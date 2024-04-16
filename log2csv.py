import argparse
import numpy as np 
import ipdb 
import os 
import csv 

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=None)
parser.add_argument('--multi', action='store_true')
args = parser.parse_args()
total_idx = [1, 2, 3]
# if args.multi:
    # total_idx = [1, 2, 3]
# else:
    # total_idx = [-1]

def generate_path_template(standard_path):
    child_dirs = standard_path.split('/')
    child_dirs.remove('')
    final_dirs = child_dirs[-1].split('_')
    final_dirs[-1] = '{}'
    # child_path = final_dirs.join('_')
    child_path = '_'.join(final_dirs)
    return os.path.join(*child_dirs[:-1], child_path)

def get_each_file(args, idx):
    res = {}
    dir_path = generate_path_template(args.path).format(idx)
    def add_element(res, name, element):
        if name not in res.keys():
            res[name] = []
        res[name].append(element)
    flag = 'mess'
    if os.path.exists(os.path.join(dir_path, 'LOG_ALL.log')):
        log_name = 'LOG_ALL.log'
    elif os.path.exists(os.path.join(dir_path, 'LOG.log')):
        log_name = 'LOG.log'
    elif os.path.exists(os.path.join(dir_path, 'log_train.txt')):
        log_name = 'log_train.txt'
    else:
        raise ValueError('pls check the log file name.')
    with open(os.path.join(dir_path, log_name), 'r') as f:
        for line in f:
            line = line.strip()
            if 'iou_thresh: 0.25' in line:
                flag = ' iou_thresh: 0.25'
            elif 'iou_thresh: 0.5' in line:
                flag = ' iou_thresh: 0.5'
            if ('eval' in line) and ('Average Precision' in line):
                name, score = line.split(':')
                name = name.replace('eval', '').replace('Average Precision', '').strip()
                score = float(score)
                add_element(res, name+' AP' + flag, score)
            elif ('eval' in line) and ('Recall' in line):
                name, score = line.split(':')
                name = name.replace('eval', '').replace('Recall', '').strip()
                score = float(score)
                add_element(res, name+' Recall' + flag, score)
            elif 'eval mAP' in line:
                name, score = line.split(':')
                score = float(score)
                add_element(res, 'mAP' + flag, score)
            elif 'eval AR' in line:
                name, score = line.split(':')
                score = float(score)
                add_element(res, 'AR' + flag, score)
    return res, dir_path

def write_each_file(res, dir_path):
    csv_head = []
    csv_head.append('name')
    for each in list(res.keys()):
        if 'iou_thresh: 0.25' in each:
            csv_head.append(each)
    for each in list(res.keys()):
        if 'iou_thresh: 0.5' in each:
            csv_head.append(each)
    if not os.path.exists(os.path.join('output', 'log.csv')):
        with open(os.path.join('output', 'log.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_head)

    with open(os.path.join('output', 'log.csv'), 'a') as f:
        csv_writer = csv.writer(f)
        row_list = []
        head_list = []
        row_list.append(dir_path)
        for key, value in res.items():
            if 'iou_thresh: 0.25' in key:
                row_list.append(str(value[-1]))
                head_list.append(key)
        for key, value in res.items():
            if 'iou_thresh: 0.5' in key:
                row_list.append(str(value[-1]))
                head_list.append(key)
        csv_writer.writerow(row_list)

all_res = []
for each in total_idx:
    res, dir_path = get_each_file(args, each)
    all_res.append(res)
average_res = {}
for key in all_res[0].keys():
    average_res[key] = []
    for i in range(len(all_res)):
        average_res[key].append(all_res[i][key][-1])
    average_res[key] = [np.mean(average_res[key])]
write_each_file(average_res, dir_path)
