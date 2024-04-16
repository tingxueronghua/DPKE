import numpy as np 
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from sunrgbd.model_util_sunrgbd import SunrgbdDatasetConfig
from sunrgbd import sunrgbd_utils
import ipdb 

DC = SunrgbdDatasetConfig()


def generate_votes(point_cloud, target_bboxes, scan_name, num_points):
    '''generate vote for each point
       Note: one point may belong to several objects if there are overlaps between these objects
       Returns:
            point_votes_mask: shape (num_points,)
            point_votes: shape (num_points, 9)
        
        adopted from https://github.com/Na-Z/SDCoT/blob/7625bc255ad757d948d1120db9ca9a8351825bd9/datasets/sunrgbd.py#L154
    '''
    point_votes = np.zeros((num_points, 10))  # 3 votes and 1 vote mask
    point_vote_idx = np.zeros((num_points)).astype(np.int32)  # in the range of [0,2]
    indices = np.arange(num_points)
    for target_box in target_bboxes:
        try:
            # Find all points in this object's OBB
            box3d_pts_3d = sunrgbd_utils.my_compute_box_3d(target_box[0:3], target_box[3:6], target_box[6])
            pc_in_box3d, inds = sunrgbd_utils.extract_pc_in_box3d(point_cloud, box3d_pts_3d)
            # Assign first dimension to indicate it is in an object box
            point_votes[inds, 0] = 1
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(target_box[0:3], 0) - pc_in_box3d[:, 0:3]
            sparse_inds = indices[inds]  # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(point_vote_idx[j] * 3 + 1):int((point_vote_idx[j] + 1) * 3 + 1)] = votes[i, :]
                # Populate votes with the first vote
                if point_vote_idx[j] == 0:
                    point_votes[j, 4:7] = votes[i, :]
                    point_votes[j, 7:10] = votes[i, :]
            point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds] + 1)
        except:
            print('ERROR ----', scan_name, target_box[7])

    return point_votes

def sample_from_base(input_dict, base, base_class_dict, sample_num_per_class=1, sample_iou_threshold=0.2):
    '''
    @input input_dict: {'point_clouds': (#points, 3+Color), 'bboxes': (N, 8), )}
    @base_class_dict: {class_index:array(index_list)}
    '''
    base_bboxes = base['instance_bboxes'] 
    base_raw_pc = base['raw_points']

    input_pc = input_dict['point_clouds']
    input_bbox = input_dict['bboxes']
    
    # this part, the last element of bboxes should be heading
    # for scanent, this heading value should be zero 
    # iou_base_bboxes = np.zeros_like(base_bboxes)
    # iou_input_bbox = np.zeros_like(input_bbox)
    iou_base_bboxes = base_bboxes[:, :7]
    iou_input_bbox = input_bbox[:, :7]
    
    categories = np.concatenate([np.arange(DC.num_class), np.arange(DC.num_class)])
    # categories = np.arange(DC.num_class)
    np.random.shuffle(categories)
    selected_indx_list = []
    for each_category in categories:
        class_idxs = base_class_dict[each_category]
        tem_idxs = np.random.choice(class_idxs, (sample_num_per_class))
        if len(selected_indx_list) == 0:
            sampled_bbox = iou_input_bbox
        else:
            selected_indx = np.concatenate(selected_indx_list, axis=0)
            sampled_bbox = np.concatenate([iou_input_bbox, iou_base_bboxes[selected_indx]], axis=0)
        iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(iou_base_bboxes[tem_idxs], sampled_bbox)
        iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(iou_base_bboxes[tem_idxs], iou_base_bboxes[tem_idxs])
        iou2[range(iou2.shape[0]), range(iou2.shape[0])] = 0
        if iou1.shape[1] > 0:
            iou1_mask = (np.max(iou1, axis=1) == 0)
            iou2_mask = (np.max(iou2, axis=1) == 0)
            mask_final = np.logical_and(iou1_mask, iou2_mask)
        else:
            iou2_mask = (np.max(iou2, axis=1) == 0)
            mask_final = iou2_mask

        indexes_final_each_category = np.where(mask_final)[0]
        selected_indx_list.append(tem_idxs[indexes_final_each_category])
    selected_indx = np.concatenate(selected_indx_list, axis=0)
    if len(selected_indx) == 0:
        return input_dict 

    # remove possible conflict point clouds 
    point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(input_pc[:, :3], iou_base_bboxes[selected_indx])
    total_point_mask = np.sum(point_masks, axis=0, dtype=np.bool_)
    input_pc = np.delete(input_pc, total_point_mask, axis=0)

    # add new points into the point clouds of input_dict 
    res = {}
    res_point_clouds = input_pc 
    res_bboxes = input_bbox 
    for indx in selected_indx:
        res_point_clouds = np.concatenate([res_point_clouds, base_raw_pc[indx]], axis=0)
        res_bboxes = np.concatenate([res_bboxes, base_bboxes[indx][np.newaxis, ...]], axis=0)
        tem_array = np.ones(base_raw_pc[indx].shape[0]) 
    res['point_clouds'] = res_point_clouds
    res['bboxes'] = res_bboxes
    return res 
