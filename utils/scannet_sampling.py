import numpy as np 
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from scannet.model_util_scannet import ScannetDatasetConfig
import ipdb 
DC = ScannetDatasetConfig()


def generate_candidate_bboxes(bboxes, floor_pos):
    '''
    generate candidate bboxes given the positions on the floor 
    @param bboxes: (N, 7), the last dimension is (xc, yc, zc, dx, dy, dz, cls)
    @param floor_pos: (M, M, 3), the last dimension is (x, y, z)
    @return res_bboxes: (N, M*M, 7)
    '''
    pass 

def sample_from_base(input_dict, base, base_class_dict, sample_num_per_class=1, sample_iou_threshold=0.2, category_list=None, category_prob=None):
    '''
    @input input_dict: {'point_clouds': (#points, 3+Color), 'bboxes': (N, 7), )}
    @base_class_dict: {class_index:array(index_list)}
    '''
    base_bboxes = base['instance_bboxes'] 
    base_raw_pc = base['raw_points']

    input_pc = input_dict['point_clouds']
    input_bbox = input_dict['bboxes']
    
    # this part, the last element of bboxes should be heading
    # for scanent, this heading value should be zero 
    iou_base_bboxes = np.zeros_like(base_bboxes)
    iou_input_bbox = np.zeros_like(input_bbox)
    iou_base_bboxes[:, :-1] = base_bboxes[:, :-1]
    iou_input_bbox[:, :-1] = input_bbox[:, :-1]
    
    if category_list is None:
        categories = np.arange(len(DC.nyu40ids))
        np.random.shuffle(categories)
    else:
        categories = category_list
    selected_indx_list = []
    for category_indx, each_category in enumerate(categories):
        if category_prob is None:
            pass 
        else:
            tem_ = np.random.uniform()
            if tem_ <= category_prob[category_indx]:
                pass 
            else:
                continue
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
    if len(selected_indx_list) == 0:
        return input_dict 
    selected_indx = np.concatenate(selected_indx_list, axis=0)

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
    if 'instance_labels' in input_dict.keys():
        input_instance_labels = input_dict['instance_labels']
        input_semantic_labels = input_dict['semantic_labels']
    
        # remove possible conflict points 
        input_instance_labels = np.delete(input_instance_labels, total_point_mask)
        input_semantic_labels = np.delete(input_semantic_labels, total_point_mask)

        res_instance_labels = input_instance_labels
        res_semantic_labels = input_semantic_labels
        for indx in selected_indx:
            tem_array = np.ones(base_raw_pc[indx].shape[0])
            res_instance_labels = np.concatenate([res_instance_labels, tem_array * np.max(res_instance_labels+1)], axis=0)
            res_semantic_labels = np.concatenate([res_semantic_labels, tem_array * base_bboxes[indx, -1]], axis=0)
        res['instance_labels'] = res_instance_labels
        res['semantic_labels'] = res_semantic_labels
    return res 

def sample_from_base_advanced(input_dict, base, use_mask=True, sample_num=3, sample_iou_threshold=0.2):
    '''
    @input input_dict: {'point_clouds': (#points, 3+Color), 'bboxes': (N, 7), 'instance_labels': (#points), 'semantic_labels': (#points)}
    '''
    # read data from base 
    base_bboxes = base['instance_bboxes'] 
    base_raw_pc = base['raw_points']

    # read input data dict
    input_pc = input_dict['point_clouds']
    input_bbox = input_dict['bboxes']
    input_instance_labels = input_dict['instance_labels']
    input_semantic_labels = input_dict['semantic_labels']

    # we try to sample the bounding boxes first, 
    # and use rotation and scale changing to augment them
    sampled_indexes = np.random.choice(np.arange(len(base_bboxes)), (sample_num), replace=False)

    # use similar strategy like height to get overall range
    # floor_height = np.percentile(input_pc[:, 2], 0.99)
    # TODO: second floor
    x_lower_threshold = np.percentile(input_pc[:, 0], 0.99)
    x_higher_threshold = np.percentile(input_pc[:, 0], 0.01)
    y_lower_threshold = np.percentile(input_pc[:, 1], 0.99)
    y_higher_threshold = np.percentile(input_pc[:, 1], 0.01)
    floor_height = np.percentile(input_pc[:, 0], 0.99)

    # generate lots of candidate bounding boxes 
    # select the one with lowest overlap with others (if not exceed the threshold)
    # different from outdoor detection in pcdet.
    # finally find a proper location for it. 
    x_array = np.arange(x_lower_threshold, x_higher_threshold)
    y_array = np.arange(y_lower_threshold, y_higher_threshold)
    candidate_x, candidate_y = np.meshgrid(x_array, y_array)
    candidate_z = np.ones_like(x_array) * floor_height
    # candidate_pos marks the position of candidate boxes on the floor 
    candidate_pos = np.stack([candidate_x, candidate_y, candidate_z], axis=-1)
    all_bboxes = [base_bboxes[i] for i in sampled_indexes]
    


    iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(base_bboxes, input_bbox)
    iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(base_bboxes, base_bboxes)
    iou2[range(iou2.shape[0]), range(iou2.shape[0])] = 0
    if iou1.shape[1] > 0:
        iou1_mask = (np.max(iou1, axis=1) == 0)
        iou2_mask = (np.max(iou2, axis=1) < sample_iou_threshold)
        mask_final = np.logical_and(iou1_mask, iou2_mask)
    else:
        iou2_mask = (np.max(iou2, axis=1) < sample_iou_threshold)
        mask_final = iou2_mask

    indexes_final = np.where(mask_final)[0]
    if len(indexes_final) == 0:
        return input_dict 
    elif len(indexes_final) < sample_num:
        pass 
    else:
        indexes_final = np.random.choice(indexes_final, (sample_num), replace=False)
    
    # organize outptu data.
    res = {}
    res_point_clouds = input_pc 
    res_bboxes = input_bbox 
    res_instance_labels = input_instance_labels 
    res_semantic_labels = input_semantic_labels
    for indx in indexes_final:
        res_point_clouds = np.concatenate([res_point_clouds, base_raw_pc[indx]], axis=0)
        res_bboxes = np.concatenate([res_bboxes, base_bboxes[indx][np.newaxis, ...]], axis=0)
        tem_array = np.ones(base_raw_pc[indx].shape[0]) 
        res_instance_labels = np.concatenate([res_instance_labels, tem_array * np.max(res_instance_labels+1)])
        res_semantic_labels = np.concatenate([res_semantic_labels, tem_array * base_bboxes[indx, -1]])
    res['point_clouds'] = res_point_clouds
    res['bboxes'] = res_bboxes
    res['instance_labels'] = res_instance_labels 
    res['semantic_labels'] = res_semantic_labels
    return res 
