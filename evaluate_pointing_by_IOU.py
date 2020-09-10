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
parser.add_argument('--IOU_file', required=True)
args = parser.parse_args()

mode = 'top'
percentage = 0.25

results = os.listdir(args.results_dir)
results_path = os.path.join(args.results_dir, 'eval_results_IOU_{}_{}.txt'.format(mode, str(100 * percentage)))
if os.path.exists(results_path):
    raise Exception('Result file already exists!')

IMAGE_SIZE = (360, 480)
type_dict = json.load(open(args.type_file, 'r'))
anno = json.load(open(args.anno, 'r'))

ious = json.load(open(args.IOU_file, 'r'))
sorted_ious = sorted(ious.items(), key=lambda kv: kv[1])

ious_for_total = {int(x[0]): x[1] for x in sorted_ious}

test_split = json.load(open('./data/splits.json', 'r'))['test']

total_ious = []
for id in test_split:
    total_ious.append((id, ious_for_total[id]))

total_ious_sorted = sorted(total_ious, key=lambda x: x[1])

count = int(len(test_split) * percentage)
if mode == 'top': # most difficult
    filtered_total = ['%06d.png' % x[0] for x in total_ious_sorted[:count]]
elif mode == 'bottom':
    filtered_total = ['%06d.png' % x[0] for x in total_ious_sorted[-count:]]

f = open(results_path, 'w')
for res in results:
    if '.txt' in res:
        continue
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

    total_result = pointing(gen_mapping, gt_mapping, type_ids=filtered_total)

    stats = [
        ('total', total_result),
    ]
    stats = OrderedDict(stats)

    message = '===================={} results===================\n'.format(res)
    for type, eval in stats.items():
        message += '{}: {}\n'.format(type.upper(), eval)
    f.write(message)

f.close()
