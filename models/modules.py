import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChangeDetectorDoubleAttDyn(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.model.change_detector.input_dim
        self.dim = cfg.model.change_detector.dim

        self.embed = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dim, kernel_size=1, padding=0),
            nn.GroupNorm(32, self.dim),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.att = nn.Conv2d(self.dim, 1, kernel_size=1, padding=0)
        self.fc1 = nn.Linear(self.input_dim // 2, 6)

    def forward(self, input_1, input_2):
        batch_size, _, H, W = input_1.size()
        input_diff = input_2 - input_1
        input_before = torch.cat([input_1, input_diff], 1)
        input_after = torch.cat([input_2, input_diff], 1)
        embed_before = self.embed(input_before)
        embed_after = self.embed(input_after)
        att_weight_before = F.sigmoid(self.att(embed_before))
        att_weight_after = F.sigmoid(self.att(embed_after))

        att_1_expand = att_weight_before.expand_as(input_1)
        attended_1 = (input_1 * att_1_expand).sum(2).sum(2)  # (batch, dim)
        att_2_expand = att_weight_after.expand_as(input_2)
        attended_2 = (input_2 * att_2_expand).sum(2).sum(2)  # (batch, dim)
        input_attended = attended_2 - attended_1
        pred = self.fc1(input_attended)

        return pred, att_weight_before, att_weight_after, attended_1, attended_2, input_attended


class AddSpatialInfo(nn.Module):

    def _create_coord(self, img_feat):
        batch_size, _, h, w = img_feat.size()
        coord_map = img_feat.new_zeros(2, h, w)
        for i in range(h):
            for j in range(w):
                coord_map[0][i][j] = (j * 2.0 / w) - 1
                coord_map[1][i][j] = (i * 2.0 / h) - 1
        sequence = [coord_map] * batch_size
        coord_map_in_batch = torch.stack(sequence)
        return coord_map_in_batch

    def forward(self, img_feat):
        coord_map = self._create_coord(img_feat)
        img_feat_aug = torch.cat([img_feat, coord_map], dim=1)
        return img_feat_aug
