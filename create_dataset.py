import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import pickle as pkl
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import ipdb
from torch import nn
from torch.utils.data import DataLoader

from models.votenet_iou_branch import VoteNet
from torch.utils.data import Dataset

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
from scannet.model_util_scannet import rotate_aligned_boxes
from scannet.model_util_scannet import ScannetDatasetConfig

from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler
from utils.tf_visualizer import Visualizer as TfVisualizer
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from models.loss_helper_labeled import get_labeled_loss
from models.loss_helper_unlabeled import get_unlabeled_loss
from models.loss_helper import get_loss


class SunrgbdDetectionVotesDataset(Dataset):
    def __init__(self,
                 split_set='train',
                 labeled_ratio=0.1,
                 labeled_sample_list=None,
                 num_points=20000,
                 use_color=False,
                 use_height=False,
                 use_v1=True,
                 augment=False,
                 scan_idx_list=None,
                 test_transductive=False):

        self.BASE_DIR = ''  # path of your sunrgbd dataset
        self.DC = SunrgbdDatasetConfig()
        assert (num_points <= 50000)
        self.use_v1 = use_v1
        if use_v1:
            self.data_path = os.path.join(
                self.BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v1_%s' % (split_set))
        else:
            self.data_path = os.path.join(
                self.BASE_DIR, 'sunrgbd_pc_bbox_votes_50k_v2_%s' % (split_set))

        self.raw_data_path = os.path.join(self.BASE_DIR, 'sunrgbd_trainval')
        self.scan_names = sorted(list(set([os.path.basename(x)[0:6] \
            for x in os.listdir(self.data_path)])))
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height

        # construct labeled and unlabeled samples for training
        if split_set == 'train':
            if test_transductive:
                if labeled_sample_list is not None:
                    labeled_scan_names = [
                        x.strip() for x in open(
                            os.path.join(self.raw_data_path,
                                         labeled_sample_list)).readlines()
                    ]
                    self.scan_names = list(
                        set(self.scan_names) - set(labeled_scan_names))
                    print('\tGet {} unlabeled scans for transductive learning'.
                          format(len(self.scan_names)))
                else:
                    print('Unknown labeled sample list: %s. Exiting...' %
                          labeled_sample_list)
                    exit(-1)
            else:
                self.labeled_ratio = labeled_ratio
                self.labeled_sample_list = labeled_sample_list
                self.get_labeled_samples()

            using_instance_segmentation = False
            self.gt_base = self.create_base(
                self.scan_names,
                using_instance_segmentation=using_instance_segmentation)
            if using_instance_segmentation:
                gt_base_filename = os.path.join(
                    self.BASE_DIR,
                    '{}_gt_base.pkl'.format(labeled_sample_list))
            else:
                gt_base_filename = os.path.join(
                    self.BASE_DIR,
                    '{}_gt_base_no_instance_segmentation.pkl'.format(
                        labeled_sample_list))

            with open(gt_base_filename, 'wb') as gt_f:
                pkl.dump(self.gt_base, gt_f)

    def get_labeled_samples(self):
        if self.labeled_sample_list is not None:
            labeled_scan_names = [
                x.strip() for x in open(
                    os.path.join(self.raw_data_path,
                                 self.labeled_sample_list)).readlines()
            ]
        else:
            # randomly select scan names w.r.t labeled_ratio
            num_scans = len(self.scan_names)
            num_labeled_scans = int(self.labeled_ratio * num_scans)
            scan2label = np.zeros((num_scans, self.DC.num_class))
            for i, scan_name in enumerate(self.scan_names):
                bboxes = np.load(
                    os.path.join(self.data_path, scan_name) +
                    '_bbox.npy')  # K,8
                class_ind = bboxes[:, -1]
                if len(class_ind) != 0:
                    unique_class_ind = np.unique(class_ind)
                else:
                    continue
                for j in unique_class_ind:
                    scan2label[i, int(j)] = 1

            while True:
                choices = np.random.choice(num_scans,
                                           num_labeled_scans,
                                           replace=False)
                class_distr = np.sum(scan2label[choices], axis=0)
                class_mask = np.where(class_distr > 0, 1, 0)
                if np.sum(class_mask) == self.DC.num_class:
                    labeled_scan_names = list(
                        np.array(self.scan_names)[choices])
                    with open(
                            os.path.join(
                                self.raw_data_path,
                                'sunrgbd_v1_train_{}.txt'.format(
                                    self.labeled_ratio)), 'w') as f:
                        for scan_name in labeled_scan_names:
                            f.write(scan_name + '\n')
                    break

        unlabeled_scan_names = list(
            set(self.scan_names) - set(labeled_scan_names))
        print('Selected {} labeled scans, remained {} unlabeled scans'.format(
            len(labeled_scan_names), len(unlabeled_scan_names)))
        self.scan_names = labeled_scan_names

    def create_base(self, scan_names, using_instance_segmentation=False):
        '''
        @scan_names: names for all the scans needed to build the data base 
        @res_base (output): the data base for all the proposals
            must contain:
            raw_points
            instance_bboxes
        '''
        res_base = {}
        mask_per_ins_list = []
        instance_boxes_all = []
        raw_points = []
        for idx, scan_name in enumerate(scan_names):
            mesh_vertices = np.load(
                os.path.join(self.data_path, scan_name) +
                '_pc.npz')['pc']  # Nx6
            instance_bboxes = np.load(
                os.path.join(self.data_path, scan_name) + '_bbox.npy')  # K,8
            point_votes = np.load(
                os.path.join(self.data_path, scan_name) +
                '_votes.npz')['point_votes']  # Nx10

            # 1. align instance and bboxes
            # there numbers are not the same.
            # (first mask) first use instance boxes and semantic labels to make sure that the selected points
            # in the corresponding instance box are in the same class.
            # (second mask) Then use instance labels to make sure that the points selected all belongs to
            # a single instance, which has the most points in the first_mask.
            # 2. not sure why, but the comments mention that input bboxes cannot overlap with each other.
            # so we use a loop
            # https://github.com/open-mmlab/OpenPCDet/blob/4713332c5b73b32ac23b425022a06861bfa23b89/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp#L144
            for i_ins, each_bbox in enumerate(instance_bboxes):
                tem_bbox = each_bbox[:-1]
                tem_bbox = tem_bbox[np.newaxis, ...]
                # semantic_mask = (semantic_labels == each_bbox[-1])
                point_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                    mesh_vertices[:, :3], tem_bbox)
                point_mask = point_mask.squeeze()
                # first_mask = np.logical_and(point_mask, semantic_mask)
                # if np.sum(first_mask) == 0:
                # continue
                if np.sum(point_mask) == 0:
                    continue
                mask_per_ins_list.append(point_mask)
                raw_points.append(mesh_vertices[np.where(point_mask)[0]])
                instance_boxes_all.append(each_bbox)
        res_base['raw_points'] = raw_points
        res_base['instance_bboxes'] = np.stack(instance_boxes_all, axis=0)
        res_base['mask_per_instance'] = mask_per_ins_list

        return res_base

    def __len__(self):
        return len(self.scan_names)


class ScannetDetectionDataset(Dataset):
    def __init__(self,
                 split_set='train',
                 labeled_ratio=0.1,
                 labeled_sample_list=None,
                 num_points=20000,
                 use_color=False,
                 use_height=False,
                 augment=False,
                 remove_obj=False,
                 test_transductive=False):

        self.BASE_DIR = ""  # path of you scannet dataset
        print('--------- DetectionDataset ', split_set,
              ' Initialization ---------')
        self.DC = ScannetDatasetConfig()
        self.data_path = os.path.join(self.BASE_DIR,
                                      'scannet_train_detection_data')
        all_scan_names = list(set([os.path.basename(x)[0:12] \
            for x in os.listdir(self.data_path) if x.startswith('scene')]))
        if split_set == 'all':
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(
                self.BASE_DIR, 'meta_data',
                'scannetv2_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('\tkept {} scans out of {}'.format(len(self.scan_names),
                                                     num_scans))
            num_scans = len(self.scan_names)
        else:
            print('\tillegal split name')
            return

        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        self.augment = augment
        self.remove_obj = remove_obj

        # added
        self.raw_data_path = os.path.join(self.BASE_DIR, 'meta_data')

        self.scans_data_path = os.path.join(self.BASE_DIR, 'scans')

        # construct labeled and unlabeled samples for training
        if split_set == 'train':
            if test_transductive:
                if labeled_sample_list is not None:
                    labeled_scan_names = [
                        x.strip() for x in open(
                            os.path.join(self.raw_data_path,
                                         labeled_sample_list)).readlines()
                    ]
                    self.scan_names = list(
                        set(self.scan_names) - set(labeled_scan_names))
                    print('\tGet {} unlabeled scans for transductive learning'.
                          format(len(self.scan_names)))
                else:
                    print('Unknown labeled sample list: %s. Exiting...' %
                          labeled_sample_list)
                    exit(-1)
            else:
                self.labeled_ratio = labeled_ratio
                self.labeled_sample_list = labeled_sample_list
                self.get_labeled_samples()

            using_instance_segmentation = False
            self.gt_base = self.create_base(
                self.scan_names,
                using_instance_segmentation=using_instance_segmentation)
            if using_instance_segmentation:
                gt_base_filename = os.path.join(
                    self.BASE_DIR,
                    '{}_gt_base.pkl'.format(labeled_sample_list))
            else:
                gt_base_filename = os.path.join(
                    self.BASE_DIR,
                    '{}_gt_base_no_instance_segmentation.pkl'.format(
                        labeled_sample_list))

            with open(gt_base_filename, 'wb') as gt_f:
                pkl.dump(self.gt_base, gt_f)

    def get_parameters(self, pcs, bboxes):
        '''
        get parameters noted as Equation 7 in Correlation Field for Boosting 3D Object Detection in Structured Scenes
        pc: a list, total length is N, each element is (#points, 3+color)
        bbox: [N, 7]
        this only works for single pc instance point clouds.
        '''
        param_s = bboxes[:, 3:6]
        param_cg = np.zeros((bboxes.shape[0], 3))

        pc_center = [np.mean(x[:, :3], axis=0) for x in pcs]
        pc_center = np.stack(pc_center, axis=0)
        param_cg = pc_center - bboxes[:, :3]
        return param_s, param_cg

    def create_base(self, scan_names, using_instance_segmentation=False):
        '''
        @scan_names: names for all the scans needed to build the data base 
        @res_base (output): the data base for all the proposals
            must contain:
            raw_points
            instance_bboxes
        '''
        res_base = {}
        mask_per_ins_list = []
        instance_boxes_all = []
        raw_points = []
        for idx, scan_name in enumerate(scan_names):
            mesh_vertices = np.load(
                os.path.join(self.data_path, scan_name) + '_vert.npy')
            instance_labels = np.load(
                os.path.join(self.data_path, scan_name) + '_ins_label.npy')
            semantic_labels = np.load(
                os.path.join(self.data_path, scan_name) + '_sem_label.npy')
            instance_bboxes = np.load(
                os.path.join(self.data_path, scan_name) + '_bbox.npy')

            meta_file = os.path.join(self.scans_data_path, scan_name,
                                     '{}.txt'.format(scan_name))
            lines = open(meta_file, 'r').readlines()
            for line in lines:
                if 'axisAlignment' in line:
                    axis_align_matrix = [float(x) \
                        for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                    break
            axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

            # 1. align instance and bboxes
            # there numbers are not the same.
            # (first mask) first use instance boxes and semantic labels to make sure that the selected points
            # in the corresponding instance box are in the same class.
            # (second mask) Then use instance labels to make sure that the points selected all belongs to
            # a single instance, which has the most points in the first_mask.
            # 2. not sure why, but the comments mention that input bboxes cannot overlap with each other.
            # so we use a loop
            # https://github.com/open-mmlab/OpenPCDet/blob/4713332c5b73b32ac23b425022a06861bfa23b89/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d.cpp#L144
            for i_ins, each_bbox in enumerate(instance_bboxes):
                tem_bbox = np.zeros_like(each_bbox)
                tem_bbox[:-1] = each_bbox[:-1]
                tem_bbox = tem_bbox[np.newaxis, ...]
                semantic_mask = (semantic_labels == each_bbox[-1])
                point_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
                    mesh_vertices[:, :3], tem_bbox)
                point_mask = point_mask.squeeze()
                first_mask = np.logical_and(point_mask, semantic_mask)
                if np.sum(first_mask) == 0:
                    continue
                if using_instance_segmentation:
                    instance_indx = np.argmax(
                        np.bincount(instance_labels[np.where(first_mask)[0]]))
                    instance_mask = (instance_labels == instance_indx)
                    second_mask = np.logical_and(first_mask, instance_mask)
                    # mask_per_ins.append(second_mask)
                    mask_per_ins_list.append(second_mask)
                    raw_points.append(mesh_vertices[np.where(second_mask)[0]])
                    instance_boxes_all.append(each_bbox)
                else:
                    mask_per_ins_list.append(point_mask)
                    raw_points.append(mesh_vertices[np.where(point_mask)[0]])
                    instance_boxes_all.append(each_bbox)
        res_base['raw_points'] = raw_points
        res_base['instance_bboxes'] = np.stack(instance_boxes_all, axis=0)
        res_base['mask_per_instance'] = mask_per_ins_list

        # add parameters
        param_s, param_cg = self.get_parameters(raw_points,
                                                res_base['instance_bboxes'])
        res_base['param_s'] = param_s
        res_base['param_cg'] = param_cg

        return res_base

    def get_labeled_samples(self):
        if self.labeled_sample_list is not None:
            labeled_scan_names = [
                x.strip() for x in open(
                    os.path.join(self.BASE_DIR, 'meta_data',
                                 self.labeled_sample_list)).readlines()
            ]
        else:
            # randomly select scan names w.r.t labeled_ratio
            num_scans = len(self.scan_names)
            num_labeled_scans = int(self.labeled_ratio * num_scans)
            scan2label = np.zeros((num_scans, self.DC.num_class))
            for i, scan_name in enumerate(self.scan_names):
                instance_bboxes = np.load(
                    os.path.join(self.data_path, scan_name) + '_bbox.npy')
                class_ind = [
                    self.DC.nyu40id2class[x] for x in instance_bboxes[:, -1]
                ]
                if class_ind != []:
                    unique_class_ind = list(set(class_ind))
                else:
                    continue
                for j in unique_class_ind:
                    scan2label[i, j] = 1

            while True:
                choices = np.random.choice(num_scans,
                                           num_labeled_scans,
                                           replace=False)
                class_distr = np.sum(scan2label[choices], axis=0)
                class_mask = np.where(class_distr > 0, 1, 0)
                if np.sum(class_mask) == self.DC.num_class:
                    labeled_scan_names = list(
                        np.array(self.scan_names)[choices])
                    with open(
                            os.path.join(
                                self.BASE_DIR,
                                'meta_data/scannetv2_train_{}.txt'.format(
                                    self.labeled_ratio)), 'w') as f:
                        for scan_name in labeled_scan_names:
                            f.write(scan_name + '\n')
                    break

        unlabeled_scan_names = list(
            set(self.scan_names) - set(labeled_scan_names))
        print(
            '\tSelected {} labeled scans, remained {} unlabeled scans'.format(
                len(labeled_scan_names), len(unlabeled_scan_names)))
        self.scan_names = labeled_scan_names
        print('first 3 scans', self.scan_names[:3])
        self.unlabelled_scan_names = unlabeled_scan_names


def create_dataset(dataset_name, labeled_list):
    NUM_POINT = 40000
    if dataset_name == 'sunrgbd':
        LABELED_DATASET = SunrgbdDetectionVotesDataset(
            labeled_sample_list=labeled_list,
            num_points=NUM_POINT,
            use_color=False,
            use_height=True)
    elif dataset_name == 'scannet':
        LABELED_DATASET = ScannetDetectionDataset(
            labeled_sample_list=labeled_list,
            num_points=NUM_POINT,
            use_color=False,
            use_height=True)
    else:
        print('Unknown dataset %s. Exiting...')
        exit(-1)


if __name__ == '__main__':
    dataset_name = 'sunrgbd'
    for ratio in [1.0]:
        for count in [0]:
            labeled_list = 'sunrgbd_v1_train_{}_{}.txt'.format(ratio, count)
            create_dataset(dataset_name, labeled_list)
            print('finished {}'.format(labeled_list))
    dataset_name = 'scannet'
    for ratio in [1.0]:
        for count in [0]:
            labeled_list = 'scannetv2_train_{}_{}.txt'.format(ratio, count)
            create_dataset(dataset_name, labeled_list)
            print('finished {}'.format(labeled_list))
