import os
import argparse
import json
import numpy as np

from scipy.misc import imresize

from utils.eval_utils import pointing
from collections import OrderedDict


def create_anno_from_bbox(bbox, img_size):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    anno = np.zeros(img_size)
    anno[y1:y2, x1:x2] = 1.0
    return anno

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
parser.add_argument('--anno', required=True)
parser.add_argument('--type_file', required=True)
args = parser.parse_args()

results = os.listdir(args.results_dir)
results_path = os.path.join(args.results_dir, 'eval_results_pointing.txt')
if os.path.exists(results_path):
    raise Exception('Result file already exists!')

IMAGE_SIZE = (360, 480)
type_dict = json.load(open(args.type_file, 'r'))
anno = json.load(open(args.anno, 'r'))

f = open(results_path, 'w')
for res in results:
    gen_map_dir = os.path.join(args.results_dir, res)
    gen_map_paths = os.listdir(gen_map_dir)
    image_ids = set()

    gt_mapping = {}
    gen_mapping = {}

    for gen_map_path in gen_map_paths:
        image_id = str(int(gen_map_path.split('_')[0]))
        image_ids.add(image_id)

    for image_id in image_ids:
        gt_bbox_before = anno[image_id]['bbox_before']
        gt_bbox_after = anno[image_id]['bbox_after']
        if gt_bbox_before:
            gt_bbox_before = create_anno_from_bbox(gt_bbox_before, IMAGE_SIZE)
        if gt_bbox_after:
            gt_bbox_after = create_anno_from_bbox(gt_bbox_after, IMAGE_SIZE)
        gen_map_before = np.load(os.path.join(gen_map_dir, '%06d_before.npy' % int(image_id)))
        gen_map_after = np.load(os.path.join(gen_map_dir, '%06d_after.npy' % int(image_id)))
        gen_map_before = imresize(np.squeeze(gen_map_before), IMAGE_SIZE, mode='F')
        gen_map_after = imresize(np.squeeze(gen_map_after), IMAGE_SIZE, mode='F')
        gt_mapping[image_id] = (gt_bbox_before, gt_bbox_after)
        gen_mapping[image_id] = (gen_map_before, gen_map_after)

    total_result = pointing(gen_mapping, gt_mapping)
    color_result = pointing(gen_mapping, gt_mapping, type_ids=type_dict['color'])
    material_result = pointing(gen_mapping, gt_mapping, type_ids=type_dict['material'])
    add_result = pointing(gen_mapping, gt_mapping, type_ids=type_dict['add'])
    drop_result = pointing(gen_mapping, gt_mapping, type_ids=type_dict['drop'])
    move_result = pointing(gen_mapping, gt_mapping, type_ids=type_dict['move'])


    stats = [
        ('total', total_result),
        ('color', color_result),
        ('material', material_result),
        ('add', add_result),
        ('drop', drop_result),
        ('move', move_result),
    ]
    stats = OrderedDict(stats)

    message = '===================={} results===================\n'.format(res)
    for type, eval in stats.items():
        message += '{}: {}\n'.format(type.upper(), eval)
    f.write(message)

f.close()
