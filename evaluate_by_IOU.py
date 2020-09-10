import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import argparse
import json

from collections import defaultdict

from utils.eval_utils import score_generation_with_ids, \
    score_generation_by_type_with_ids, \
    coco_gen_format_save

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', required=True)
parser.add_argument('--anno', required=True)
parser.add_argument('--type_file', required=True)
parser.add_argument('--IOU_file', required=True)
args = parser.parse_args()

################################################################################
# Modify this to change percentage and difficulty
mode = 'top' # 'bottom'
percentage = 0.25
################################################################################

results = os.listdir(args.results_dir)
results_path = os.path.join(args.results_dir, 'eval_results_IOU_{}_{}.txt'.format(mode, str(100 * percentage)))
if os.path.exists(results_path):
    raise Exception('Result file already exists!')

sc_best_results = defaultdict(lambda : ('iter', -10000))

ious = json.load(open(args.IOU_file, 'r'))
sorted_ious = sorted(ious.items(), key=lambda kv: kv[1])

all_ious = {'%06d.png' % int(x[0]): x[1] for x in sorted_ious}

test_split = json.load(open('./data/splits.json', 'r'))['test']
test_ids_for_sc = ['%06d.png' % x for x in test_split]

sc_ious = []
for id in test_ids_for_sc:
    sc_ious.append((id, all_ious[id]))

sc_ious_sorted = sorted(sc_ious, key=lambda x: x[1])

count = int(len(test_split) * percentage)
if mode == 'top': # most difficult
    filtered = [x[0] for x in sc_ious_sorted[:count]]
elif mode == 'bottom': # least difficult
    filtered = [x[0] for x in sc_ious_sorted[-count:]]

f = open(results_path, 'w')
for res in results:
    if '.txt' in res:
        continue
    path = os.path.join(args.results_dir, res)
    sc_path = os.path.join(path, 'sc_results.json')
    sc_captions = json.load(open(sc_path, 'r'))
    sc_eval_result = score_generation_with_ids(args.anno, sc_path, filtered)
    message = '===================={} results===================\n'.format(res)
    message += '-------------semantic change captions only----------\n'
    for k, v in sc_eval_result.items():
        iter_name , prev_best = sc_best_results[k]
        if prev_best < v:
            sc_best_results[k] = (res, v)
        message += '{}: {}\n'.format(k, v)
    f.write(message)
f.close()
