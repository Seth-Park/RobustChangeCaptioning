from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from .CaptionModel import CaptionModel


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = inv_ix.new_tensor(torch.arange(0, len(indices)))
    return tmp, inv_ix


def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp


def last_timestep(unpacked, lengths):
    # Index of the last output for each sequence.
    idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0),
                                           unpacked.size(2)).unsqueeze(1)
    return unpacked.gather(1, idx).squeeze()


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)


class DynamicCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.rnn_num_layers = 2
        self.drop_prob_lm = cfg.model.speaker.drop_prob_lm
        self.input_dim = cfg.model.speaker.input_dim
        self.embed_input_dim = cfg.model.speaker.embed_input_dim
        self.embed_dim = cfg.model.speaker.embed_dim

        self.embed = nn.Sequential(
            nn.Linear(self.embed_input_dim, self.embed_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm)
        )

        self.module_att_lstm = nn.LSTMCell(self.embed_dim + cfg.model.speaker.rnn_size,
                                           cfg.model.speaker.rnn_size)

        self.weight_fc = nn.Sequential(
            nn.Linear(cfg.model.speaker.rnn_size, 3),
            nn.Softmax(dim=1)
        )

        self.lang_lstm = nn.LSTMCell(cfg.model.speaker.word_embed_size + \
                                     cfg.model.speaker.input_dim,
                                     cfg.model.speaker.rnn_size)
        self.module_weights = None


    def forward(self, xt,
                loc_feat_bef,loc_feat_aft,
                feat_diff, state):

        prev_h = state[0][-1]  # prev hidden state from the lang_lstm
        embed_input = torch.cat([loc_feat_bef, feat_diff, loc_feat_aft], 1)
        embed = self.embed(embed_input)
        module_att_lstm_input = torch.cat([embed, prev_h], 1)

        h_mod_att, c_mod_att = self.module_att_lstm(module_att_lstm_input, (state[0][0], state[1][0]))
        module_weights = self.weight_fc(h_mod_att)
        self.module_weights = module_weights

        feats = torch.cat([loc_feat_bef.unsqueeze(1),
                           feat_diff.unsqueeze(1),
                           loc_feat_aft.unsqueeze(1)], 1)

        weights_expand = module_weights.unsqueeze(2).expand_as(feats)
        att_feat = (feats * weights_expand).sum(1)  # (batch, feat_dim)

        lang_lstm_input = torch.cat([xt, att_feat], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_mod_att, h_lang]), torch.stack([c_mod_att, c_lang]))

        return output, state


    def get_module_weights(self):
        # needs to be called after forward call
        return self.module_weights


class DynamicSpeaker(CaptionModel):
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = cfg.model.speaker.vocab_size
        self.word_embed_size = cfg.model.speaker.word_embed_size
        self.rnn_size = cfg.model.speaker.rnn_size
        self.drop_prob_lm = cfg.model.speaker.drop_prob_lm
        self.seq_length = cfg.model.speaker.seq_length

        self.ss_prob = 0.0  # Scheduled sampling probability

        self.embed = nn.Sequential(
            nn.Embedding(self.vocab_size, self.word_embed_size),
            nn.ReLU(),
            nn.Dropout(self.drop_prob_lm))

        self.core = DynamicCore(cfg)
        self.rnn_num_layers = self.core.rnn_num_layers

        self.logit_layers = getattr(cfg.model.speaker, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size),
                           nn.ReLU(),
                           nn.Dropout(0.5)] \
                          for _ in range(cfg.model.speaker.logit_layers - 1)]
            self.logit = nn.Sequential(*(
                    reduce(lambda x, y: x + y, self.logit) + \
                    [nn.Linear(self.rnn_size, self.vocab_size)]))

        self.module_weights = []


    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.rnn_num_layers, batch_size, self.rnn_size),
                weight.new_zeros(self.rnn_num_layers, batch_size, self.rnn_size))


    def _forward(self,
                 feat_bef, feat_aft,
                 feat_diff, seq):

        # start fresh
        self.module_weights = []

        batch_size = feat_bef.size(0)
        state = self.init_hidden(batch_size)

        # outputs are logprobs
        outputs = feat_bef.new_zeros(batch_size, self.seq_length, self.vocab_size)

        for i in range(self.seq_length):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = feat_bef.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)  # idx for sampling from batch
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i - 1].detach())  # fetch previous distribution (N x (M+1))
                    it.index_copy_(0, sample_ind,
                                   torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it,
                                                    feat_bef, feat_aft,
                                                    feat_diff, state)
            outputs[:, i] = output

        return outputs


    def get_logprobs_state(self, it,
                           feat_bef, feat_aft,
                           feat_diff, state):

        # 'it' contains word indices
        xt = self.embed(it)

        output, state = self.core(xt,
                                  feat_bef, feat_aft,
                                  feat_diff, state)

        # after every call of the core, collect the module weights
        self.module_weights.append(self.core.get_module_weights())
        log_probs = F.log_softmax(self.logit(output), dim=1)
        return log_probs, state


    def get_module_weights(self):
        if len(self.module_weights) == 0:
            print('no module weights accumulated')
            return None
        module_weights_stacked = torch.stack(self.module_weights, dim=1)
        return module_weights_stacked


    def _sample_beam(self,
                     feat_bef, feat_aft,
                     feat_diff, cfg={}):
        # start fresh
        self.module_weights = []

        beam_size = cfg.model.speaker.get('beam_size', 10)
        batch_size = feat_bef.size(0)

        assert beam_size <= self.vocab_size, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seq_logprobs = torch.FloatTensor(self.seq_length, batch_size)

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_feat_bef = feat_bef[k:k + 1].expand(beam_size, -1).contiguous()
            tmp_feat_aft = feat_aft[k:k + 1].expand(beam_size, -1).contiguous()
            tmp_feat_diff = feat_diff[k:k + 1].expand(beam_size, -1).contiguous()

            # input <bos> (idx of 2)
            it = feat_bef.new_zeros([beam_size], dtype=torch.long) + 2
            logprobs, state = self.get_logprobs_state(it,
                                                      tmp_feat_bef, tmp_feat_aft,
                                                      tmp_feat_diff, state)

            self.done_beams[k] = self.beam_search(state, logprobs,
                                                  tmp_feat_bef, tmp_feat_aft,
                                                  tmp_feat_diff, state, cfg=cfg)
            seq[:, k] = self.done_beams[k][0]['seq']
            seq_logprobs[:, k] = self.done_beams[k][0]['logps']

        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seq_logprobs.transpose(0, 1)


    def _sample(self,
                feat_bef, feat_aft,
                feat_diff, seq, cfg={}, sample_max=0):

        # start fresh
        self.module_weights = []

        #sample_max = cfg.model.speaker.get('sample_max', 1)
        beam_size = cfg.model.speaker.get('beam_size', 1)
        temperature = cfg.model.speaker.get('temperature', 1.0)
        decoding_constraint = cfg.model.speaker.get('decoding_contraint', 0)

        if beam_size > 1:
            return self._sample_beam(feat_bef, feat_aft,
                                     feat_diff, cfg)

        batch_size = feat_bef.size(0)
        state = self.init_hidden(batch_size)

        seq = feat_bef.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seq_logprobs = feat_bef.new_zeros(batch_size, self.seq_length)

        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = feat_bef.new_zeros(batch_size, dtype=torch.long) + 2

            logprobs, state = self.get_logprobs_state(it,
                                                      feat_bef, feat_aft,
                                                      feat_diff, state)

            # if first step, make sure we don't sample NULL
            if t == 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp[:, 0] = float('-inf')
                logprobs = logprobs + tmp
            # decoding constraint for not sampling the word sampled at t-1
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length
                break
            if sample_max:
                sample_logprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
                it = it.view(-1).long()  # and flatten indices for downstream processing

            # stop when all finished
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            it = it * unfinished.type_as(it)
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs.view(-1)
            # quit loop if all sequences have finished
            if unfinished.sum() == 0:
                break

        return seq, seq_logprobs

