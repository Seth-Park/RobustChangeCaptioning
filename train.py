import os
import sys
import json
import argparse
import time
import numpy as np
import torch
torch.backends.cudnn.enabled  = True
import torch.nn as nn
import torch.nn.functional as F

from configs.config import cfg, merge_cfg_from_file
from datasets.datasets import create_dataset
from models.modules import ChangeDetectorDoubleAttDyn, AddSpatialInfo
from models.dynamic_speaker import DynamicSpeaker
from utils.logger import Logger
from utils.utils import AverageMeter, accuracy, set_mode, save_checkpoint, \
                        LanguageModelCriterion, decode_sequence, decode_beams, \
                        build_optimizer, coco_gen_format_save, one_hot_encode, \
                        EntropyLoss

from utils.vis_utils import visualize_att

# Load config
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy_weight', type=float, default=0.0)
parser.add_argument('--visualize_every', type=int, default=10)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.exp_name == os.path.basename(args.cfg).replace('.yaml', '')

# Device configuration
use_cuda = torch.cuda.is_available()
gpu_ids = cfg.gpu_id
torch.backends.cudnn.enabled  = True
default_gpu_device = gpu_ids[0]
torch.cuda.set_device(default_gpu_device)
device = torch.device("cuda" if use_cuda else "cpu")

# Experiment configuration
exp_dir = cfg.exp_dir
exp_name = cfg.exp_name
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

output_dir = os.path.join(exp_dir, exp_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cfg_file_save = os.path.join(output_dir, 'cfg.json')
json.dump(cfg, open(cfg_file_save, 'w'))

sample_dir = os.path.join(output_dir, 'eval_gen_samples')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
sample_subdir_format = '%s_samples_%d'

sent_dir = os.path.join(output_dir, 'eval_sents')
if not os.path.exists(sent_dir):
    os.makedirs(sent_dir)
sent_subdir_format = '%s_sents_%d'

snapshot_dir = os.path.join(output_dir, 'snapshots')
if not os.path.exists(snapshot_dir):
    os.makedirs(snapshot_dir)
snapshot_file_format = '%s_checkpoint_%d.pt'

train_logger = Logger(cfg, output_dir, is_train=True)
val_logger = Logger(cfg, output_dir, is_train=False)

# Create model
change_detector = ChangeDetectorDoubleAttDyn(cfg)
change_detector.to(device)

speaker = DynamicSpeaker(cfg)
speaker.to(device)

spatial_info = AddSpatialInfo()
spatial_info.to(device)

print(change_detector)
print(speaker)
print(spatial_info)

with open(os.path.join(output_dir, 'model_print'), 'w') as f:
    print(change_detector, file=f)
    print(speaker, file=f)
    print(spatial_info, file=f)

# Data loading part
train_dataset, train_loader = create_dataset(cfg, 'train')
val_dataset, val_loader = create_dataset(cfg, 'val')
train_size = len(train_dataset)
val_size = len(val_dataset)

# Define loss function and optimizer
lang_criterion = LanguageModelCriterion().to(device)
entropy_criterion = EntropyLoss().to(device)
all_params = list(change_detector.parameters()) + list(speaker.parameters())
optimizer = build_optimizer(all_params, cfg)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=cfg.train.optim.step_size,
    gamma=cfg.train.optim.gamma)

# Train loop
t = 0
epoch = 0

set_mode('train', [change_detector, speaker])
ss_prob = speaker.ss_prob

while t < cfg.train.max_iter:
    epoch += 1
    print('Starting epoch %d' % epoch)
    lr_scheduler.step()
    speaker_loss_avg = AverageMeter()
    total_loss_avg = AverageMeter()
    if epoch > cfg.train.scheduled_sampling_start and cfg.train.scheduled_sampling_start >= 0:
        frac = (epoch - cfg.train.scheduled_sampling_start) // cfg.train.scheduled_sampling_increase_every
        ss_prob_prev = ss_prob
        ss_prob = min(cfg.train.scheduled_sampling_increase_prob * frac,
                      cfg.train.scheduled_sampling_max_prob)
        speaker.ss_prob = ss_prob
        if ss_prob_prev != ss_prob:
            print('Updating scheduled sampling rate: %.4f -> %.4f' % (ss_prob_prev, ss_prob))
    for i, batch in enumerate(train_loader):
        iter_start_time = time.time()

        d_feats, nsc_feats, sc_feats, \
        labels, no_chg_labels, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
        d_img_paths, nsc_img_paths, sc_img_paths = batch

        batch_size = d_feats.size(0)
        labels = labels.squeeze(1)
        no_chg_labels = no_chg_labels.squeeze(1)
        masks = masks.squeeze(1).float()
        no_chg_masks = no_chg_masks.squeeze(1).float()

        d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
        d_feats, nsc_feats, sc_feats = \
            spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
        labels, masks = labels.to(device), masks.to(device)
        no_chg_labels, no_chg_masks = no_chg_labels.to(device), no_chg_masks.to(device)
        aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

        optimizer.zero_grad()

        chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
        chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats)
        chg_neg_logits, chg_neg_att_bef, chg_neg_att_aft, \
        chg_neg_feat_bef, chg_neg_feat_aft, chg_neg_feat_diff = change_detector(d_feats, nsc_feats)


        speaker_output_pos = speaker._forward(chg_pos_feat_bef,
                                              chg_pos_feat_aft,
                                              chg_pos_feat_diff,
                                              labels)
        dynamic_atts = speaker.get_module_weights() # (batch, seq_len, 3)

        speaker_output_neg = speaker._forward(chg_neg_feat_bef,
                                              chg_neg_feat_aft,
                                              chg_neg_feat_diff,
                                              no_chg_labels)

        speaker_loss = 0.5 * lang_criterion(speaker_output_pos, labels[:, 1:], masks[:, 1:]) + \
                       0.5 * lang_criterion(speaker_output_neg, no_chg_labels[:, 1:], no_chg_masks[:, 1:])
        speaker_loss_val = speaker_loss.item()


        entropy_loss = -args.entropy_weight * entropy_criterion(dynamic_atts, masks[:, 1:])
        att_sum = (chg_pos_att_bef.sum() + chg_pos_att_aft.sum()) / (2 * batch_size)
        total_loss = speaker_loss + 2.5e-03 * att_sum + entropy_loss
        total_loss_val = total_loss.item()

        speaker_loss_avg.update(speaker_loss_val, 2 * batch_size)
        total_loss_avg.update(total_loss_val, 2 * batch_size)

        stats = {}
        stats['entropy_loss'] = entropy_loss.item()
        stats['speaker_loss'] = speaker_loss_val
        stats['avg_speaker_loss'] = speaker_loss_avg.avg
        stats['total_loss'] = total_loss_val
        stats['avg_total_loss'] = total_loss_avg.avg


        #results, sample_logprobs = model(d_feats, q_feats, labels, cfg=cfg, mode='sample')
        total_loss.backward()
        optimizer.step()

        iter_end_time = time.time() - iter_start_time

        t += 1

        if t % cfg.train.log_interval == 0:
            train_logger.print_current_stats(epoch, i, t, stats, iter_end_time)
            train_logger.plot_current_stats(
                epoch,
                float(i * batch_size) / train_size, stats, 'loss')

        if t % cfg.train.snapshot_interval == 0:
            speaker_state = speaker.state_dict()
            chg_det_state = change_detector.state_dict()
            checkpoint = {
                'change_detector_state': chg_det_state,
                'speaker_state': speaker_state,
                'model_cfg': cfg
            }
            save_path = os.path.join(snapshot_dir,
                                     snapshot_file_format % (exp_name, t))
            save_checkpoint(checkpoint, save_path)

            print('Running eval at iter %d' % t)
            set_mode('eval', [change_detector, speaker])
            with torch.no_grad():
                test_iter_start_time = time.time()

                idx_to_word = train_dataset.get_idx_to_word()

                if args.visualize:
                    sample_subdir_path = sample_subdir_format % (exp_name, t)
                    sample_save_dir = os.path.join(sample_dir, sample_subdir_path)
                    if not os.path.exists(sample_save_dir):
                        os.makedirs(sample_save_dir)
                sent_subdir_path = sent_subdir_format % (exp_name, t)
                sent_save_dir = os.path.join(sent_dir, sent_subdir_path)
                if not os.path.exists(sent_save_dir):
                    os.makedirs(sent_save_dir)


                result_sents_pos = {}
                result_sents_neg = {}
                for val_i, val_batch in enumerate(val_loader):
                    d_feats, nsc_feats, sc_feats, \
                    labels, no_chg_labels, masks, no_chg_masks, aux_labels_pos, aux_labels_neg, \
                    d_img_paths, nsc_img_paths, sc_img_paths = val_batch

                    val_batch_size = d_feats.size(0)

                    d_feats, nsc_feats, sc_feats = d_feats.to(device), nsc_feats.to(device), sc_feats.to(device)
                    d_feats, nsc_feats, sc_feats = \
                        spatial_info(d_feats), spatial_info(nsc_feats), spatial_info(sc_feats)
                    labels, masks = labels.to(device), masks.to(device)
                    no_chg_labels, no_chg_masks = no_chg_labels.to(device), no_chg_masks.to(device)
                    aux_labels_pos, aux_labels_neg = aux_labels_pos.to(device), aux_labels_neg.to(device)

                    chg_pos_logits, chg_pos_att_bef, chg_pos_att_aft, \
                    chg_pos_feat_bef, chg_pos_feat_aft, chg_pos_feat_diff = change_detector(d_feats, sc_feats)
                    chg_neg_logits, chg_neg_att_bef, chg_neg_att_aft, \
                    chg_neg_feat_bef, chg_neg_feat_aft, chg_neg_feat_diff = change_detector(d_feats, nsc_feats)


                    speaker_output_pos, _ = speaker._sample(chg_pos_feat_bef,
                                                            chg_pos_feat_aft,
                                                            chg_pos_feat_diff,
                                                            labels, cfg)

                    pos_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy() # (batch, seq_len, 3)

                    speaker_output_neg, _ = speaker._sample(chg_neg_feat_bef,
                                                            chg_neg_feat_aft,
                                                            chg_neg_feat_diff,
                                                            no_chg_labels, cfg)

                    neg_dynamic_atts = speaker.get_module_weights().detach().cpu().numpy() # (batch, seq_len, 3)

                    gen_sents_pos = decode_sequence(idx_to_word, speaker_output_pos)
                    gen_sents_neg = decode_sequence(idx_to_word, speaker_output_neg)

                    chg_pos_att_bef = chg_pos_att_bef.cpu().numpy()
                    chg_pos_att_aft = chg_pos_att_aft.cpu().numpy()

                    chg_neg_att_bef = chg_neg_att_bef.cpu().numpy()
                    chg_neg_att_aft = chg_neg_att_aft.cpu().numpy()
                    dummy = np.ones_like(chg_pos_att_bef)

                    for val_j in range(speaker_output_pos.size(0)):
                        gts = decode_sequence(idx_to_word, labels[val_j][:, 1:])
                        gts_neg = decode_sequence(idx_to_word, no_chg_labels[val_j][:, 1:])
                        if args.visualize and val_j % args.visualize_every == 0:
                            visualize_att(d_img_paths[val_j], sc_img_paths[val_j],
                                          chg_pos_att_bef[val_j], dummy[val_j], chg_pos_att_aft[val_j],
                                          pos_dynamic_atts[val_j], gen_sents_pos[val_j], gts,
                                          -1, -1, sample_save_dir, 'sc_')
                            visualize_att(d_img_paths[val_j], nsc_img_paths[val_j],
                                          chg_neg_att_bef[val_j], dummy[val_j], chg_neg_att_aft[val_j],
                                          neg_dynamic_atts[val_j], gen_sents_neg[val_j], gts_neg,
                                          -1, -1, sample_save_dir, 'nsc_')
                        sent_pos = gen_sents_pos[val_j]
                        sent_neg = gen_sents_neg[val_j]
                        image_id = d_img_paths[val_j].split('_')[-1]
                        result_sents_pos[image_id] = sent_pos
                        result_sents_neg[image_id + '_n'] = sent_neg
                        message = '%s results:\n' % d_img_paths[val_j]
                        message += '\t' + sent_pos + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts:
                            message += gt + '\n'
                        message += '===================================\n'
                        message += '%s results:\n' % nsc_img_paths[val_j]
                        message += '\t' + sent_neg + '\n'
                        message += '----------<GROUND TRUTHS>----------\n'
                        for gt in gts_neg:
                            message += gt + '\n'
                        message += '===================================\n'
                        print(message)


                test_iter_end_time = time.time() - test_iter_start_time
                result_save_path_pos = os.path.join(sent_save_dir, 'sc_results.json')
                result_save_path_neg = os.path.join(sent_save_dir, 'nsc_results.json')
                coco_gen_format_save(result_sents_pos, result_save_path_pos)
                coco_gen_format_save(result_sents_neg, result_save_path_neg)

            set_mode('train', [change_detector, speaker])
