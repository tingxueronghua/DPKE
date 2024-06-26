# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

from models.loss_helper_iou import compute_iou_labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.nn_distance import nn_distance, huber_loss

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from models.ap_helper import predictions2corners3d
from pointnet2 import pointnet2_utils

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

DISTANCE_MAX_THRESHOLD = 1000000

def get_chamfer_weight(input_points_original, objectness_mask, end_points, ema_end_points_static, config_dict, sample_num=500):
    student_bboxes_corner, student_bboxes = predictions2corners3d(end_points=end_points, config_dict=config_dict)
    teacher_bboxes_corner, teacher_bboxes = predictions2corners3d(end_points=ema_end_points_static, config_dict=config_dict)

    student_bboxes = torch.from_numpy(student_bboxes).to(input_points_original.device)
    teacher_bboxes = torch.from_numpy(teacher_bboxes).to(input_points_original.device)

    input_points = input_points_original[:, :, :3]
    student_point_mask = roiaware_pool3d_utils.points_in_boxes_gpu(input_points, student_bboxes)
    teacher_point_mask = roiaware_pool3d_utils.points_in_boxes_gpu(input_points, teacher_bboxes)
    bsize, num_proposal = student_bboxes.shape[0], student_bboxes.shape[1]
    
    final_weight = torch.zeros((bsize, num_proposal)).to(input_points.device)
    student_empty_mask = torch.zeros((bsize, num_proposal)).to(input_points.device).bool()
    teacher_empty_mask = torch.zeros((bsize, num_proposal)).to(input_points.device).bool()
    for i in range(bsize):
        each_objectness_mask = objectness_mask[i] 
        each_student_pc = torch.zeros((num_proposal, sample_num, 3)).to(input_points.device)
        each_teacher_pc = torch.zeros((num_proposal, sample_num, 3)).to(input_points.device)
        for j in range(num_proposal):
            mask_num = (student_point_mask[i] == j).sum()
            if mask_num <  sample_num and mask_num > 0:
                student_pc = input_points[i][student_point_mask[i] == j]
                original_inds = torch.arange(0, len(student_pc))
                new_inds = torch.randint(0, len(student_pc), size=(sample_num - len(student_pc), ))
                sample_inds = torch.cat([original_inds, new_inds], dim=0).to(input_points.device).long()
                each_student_pc[j] = student_pc[sample_inds]
            elif mask_num == 0:
                student_empty_mask[i, j] = True
                each_student_pc[j] = -DISTANCE_MAX_THRESHOLD * torch.ones_like(each_student_pc[j])
            else:
                student_pc = input_points[i][student_point_mask[i] == j]
                sample_inds = pointnet2_utils.furthest_point_sample(student_pc.unsqueeze(0), sample_num).squeeze().long()
                each_student_pc[j] = student_pc[sample_inds]
        for j in range(num_proposal):
            mask_num = (teacher_point_mask[i] == j).sum()
            if mask_num <  sample_num and mask_num > 0:
                teacher_pc = input_points[i][teacher_point_mask[i] == j]
                original_inds = torch.arange(0, len(teacher_pc))
                new_inds = torch.randint(0, len(teacher_pc), size=(sample_num - len(teacher_pc), ))
                sample_inds = torch.cat([original_inds, new_inds], dim=0).to(input_points.device).long()
                each_teacher_pc[j] = teacher_pc[sample_inds]
            elif mask_num == 0:
                teacher_empty_mask[i, j] = True
                each_teacher_pc[j] = -DISTANCE_MAX_THRESHOLD * torch.ones_like(each_teacher_pc[j])
            else:
                teacher_pc = input_points[i][teacher_point_mask[i] == j]
                sample_inds = pointnet2_utils.furthest_point_sample(teacher_pc.unsqueeze(0), sample_num).squeeze().long()
                each_teacher_pc[j] = teacher_pc[sample_inds]
        each_distance = torch.norm(each_student_pc.unsqueeze(2) - each_teacher_pc.unsqueeze(1), dim=-1)
        student_min_diff, _ = torch.min(each_distance, dim=2)
        teacher_min_diff, _ = torch.min(each_distance, dim=1)
        student_min_diff[student_empty_mask[i]] = DISTANCE_MAX_THRESHOLD 
        teacher_min_diff[teacher_empty_mask[i]] = DISTANCE_MAX_THRESHOLD
        total_diff = student_min_diff.mean(dim=1) + teacher_min_diff.mean(dim=1)
        weight_temp = total_diff[total_diff < DISTANCE_MAX_THRESHOLD].max() / np.log(2)
        # final_weight[i] = student_min_diff.mean(dim=1) + teacher_min_diff.mean(dim=1)
        final_weight[i] = torch.exp(-total_diff / (weight_temp + 1e-5))
    final_weight = final_weight.clamp(min=0.1)
    return final_weight

def set_flooding(x, thres=0.0001):
    return torch.abs(x - thres) + thres


def compute_feature_consistency_loss(end_points, ema_end_points, cfg, EPOCH_CNT):
    input_points = end_points['point_clouds']
    teacher_conv0 = ema_end_points['aggregated_vote_features'].detach()
    student_conv0 = end_points['aggregated_vote_features']

    teacher_conv1 = ema_end_points['conv1_features'].detach() 
    teacher_conv2 = ema_end_points['conv2_features'].detach()
    student_conv1 = end_points['conv1_features']
    student_conv2 = end_points['conv2_features']


    if cfg['align_features'] == 'conv2':
        student_features = student_conv2
        teacher_features = teacher_conv2
    elif cfg['align_features'] == 'conv0':
        student_features = student_conv0
        teacher_features = teacher_conv0
    elif cfg['align_features'] == 'all':
        student_features = torch.cat([student_conv0, student_conv1, student_conv2], dim=-1)
        teacher_features = torch.cat([teacher_conv0, teacher_conv1, teacher_conv2], dim=-1)
    else:
        raise ValueError('no such type')


    objectness_scores = ema_end_points['objectness_scores']
    objectness_scores = nn.functional.softmax(objectness_scores, dim=2)[:, :, 1]
    objectness_mask = objectness_scores > cfg['features_consistency_obj_threshold'] 

    supervised_mask = end_points['supervised_mask'].bool() 
    unsupervised_mask = ~supervised_mask


    final_mask = objectness_mask * unsupervised_mask.unsqueeze(-1)

    chamfer_weight = get_chamfer_weight(input_points, objectness_mask, end_points, ema_end_points, cfg) 

    element_loss = nn.functional.huber_loss(student_features, teacher_features, reduction='none', delta=1.0)
    end_points['features_element_huber'] = element_loss
    end_points['features_objectness_mask'] = objectness_mask
    loss = final_mask.unsqueeze(-1) * element_loss * chamfer_weight.unsqueeze(-1)
    if EPOCH_CNT < 500:
        loss = set_flooding(loss)
    loss = torch.mean(loss)
    end_points['features_huber_loss'] = loss 
    return loss 


def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    # print(end_points['center_label'].shape)
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config, test_time=False):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)
    end_points['cls_acc'] = torch.sum(
        (sem_cls_label == end_points['sem_cls_scores'].argmax(dim=-1))).float() / sem_cls_label.view(-1).shape[0]
    end_points['cls_acc_obj'] = torch.sum(
        (sem_cls_label == end_points['sem_cls_scores'].argmax(dim=-1)) * objectness_label) / (
                                    torch.sum(objectness_label) + 1e-6)

    # end_points['center'].retain_grad()
    mask = torch.arange(batch_size).cuda()
    iou_labels, iou_zero_mask, _ = compute_iou_labels(
        end_points, mask, end_points['aggregated_vote_xyz'], end_points['center'], None, None,
        end_points['heading_scores'],
        end_points['heading_residuals'],
        end_points['size_scores'],
        end_points['size_residuals'], {'dataset_config': config})

    end_points['iou_labels'] = iou_labels
    end_points['pred_iou_value'] = torch.sum(iou_labels) / iou_labels.view(-1).shape[0]
    end_points['pred_iou_obj_value'] = torch.sum(iou_labels * objectness_label) / (torch.sum(objectness_label) + 1e-6)

    if 'iou_scores' in end_points.keys():
        iou_pred = nn.Sigmoid()(end_points['iou_scores'])
        if iou_pred.shape[2] > 1:
            iou_pred = torch.gather(iou_pred, 2, end_points['sem_cls_scores'].argmax(dim=-1).unsqueeze(-1)).squeeze(-1)  # use pred semantic labels
        else:
            iou_pred = iou_pred.squeeze(-1)
        iou_acc = torch.abs(iou_pred - iou_labels)
        end_points['iou_acc'] = torch.sum(iou_acc) / torch.sum(torch.ones(iou_acc.shape))
        end_points['iou_acc_obj'] = torch.sum(iou_acc * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        iou_loss = huber_loss(iou_pred - iou_labels, delta=1.0)  # (B, K, 1)
        iou_loss = torch.sum(iou_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        end_points['iou_loss'] = iou_loss

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss

def get_loss(end_points, config, test_time=False):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config, test_time=test_time)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss + center_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    if 'iou_loss' in end_points.keys():
        loss = loss + end_points['iou_loss']

    loss *= 10
    end_points['detection_loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_scores = end_points['objectness_scores']
    obj_pred_val = torch.argmax(obj_scores, 2)  # B,K
    obj_acc = torch.sum((obj_pred_val == objectness_label.long()).float() * objectness_mask) / (
            torch.sum(objectness_mask) + 1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points
