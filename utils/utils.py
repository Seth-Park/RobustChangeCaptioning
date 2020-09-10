import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json


################################################################################
# Checkpoint related util functions
################################################################################
def load_cpu(path):
    """
    Loads a torch checkpoint, remapping all Tensors to CPU
    """
    return torch.load(path, map_location=lambda storage, loc: storage)

def save_checkpoint(checkpoint, filename):
    print('Saving checkpoint to %s' % filename)
    torch.save(checkpoint, filename)

def load_checkpoint(filename):
    print('Loading checkpoint from %s' % filename)
    return load_cpu(filename)

################################################################################
# Metric related util functions
################################################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

################################################################################
# Training related util functions
################################################################################
def adjust_learning_rate(base_lr, step_ratio, optimizer, curr_epoch, decay_freq):
    """Sets the learning rate accordingly to the decay schedule"""
    lr = base_lr * (step_ratio ** (curr_epoch // decay_freq))
    print('Epoch [{}] Learning rate: {}'.format(curr_epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_mode(mode, models):
    assert mode in ['train', 'eval']
    for m in models:
        if m is None: continue
        if mode == 'train': m.train()
        if mode == 'eval': m.eval()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def build_optimizer(params, cfg):
    if cfg.train.optim.type == 'rmsprop':
        return optim.RMSprop(params, cfg.train.optim.lr, cfg.train.optim.alpha, \
                             cfg.train.optim.epsilon, weight_decay=cfg.train.optim.weight_decay)
    elif cfg.train.optim.type == 'adagrad':
        return optim.Adagrad(params, cfg.train.optim.lr, weight_decay=cfg.train.optim.weight_decay)
    elif cfg.train.optim.type == 'sgd':
        return optim.SGD(params, cfg.train.optim.lr, weight_decay=cfg.train.optim.weight_decay)
    elif cfg.train.optim.type == 'sgdm':
        return optim.SGD(params, cfg.train.optim.lr, cfg.train.optim.alpha, \
                         weight_decay=cfg.train.optim.weight_decay)
    elif cfg.train.optim.type == 'sgdmom':
        return optim.SGD(params, cfg.train.optim.lr, cfg.train.optim.alpha, \
                         weight_decay=cfg.train.optim.weight_decay, nesterov=True)
    elif cfg.train.optim.type == 'adam':
        return optim.Adam(params, cfg.train.optim.lr, \
                          (cfg.train.optim.alpha, cfg.train.optim.beta), \
                          cfg.train.optim.epsilon, weight_decay=cfg.train.optim.weight_decay)
    else:
        raise Exception("bad option for optimizer: {}".format(cfg.train.optim.type))
    
################################################################################
# Language related util functions
################################################################################
# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is NULL token.
def get_sequence_length(seq):
    N, D = seq.size()
    lengths = []
    for i in range(N):
        length = 0
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
                length += 1
            else:
                break
        lengths.append(length)

    return lengths

def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix.item()]
            else:
                break
        out.append(txt)
    return out

def decode_sequence_np(ix_to_word, seq):
    N, D = seq.shape
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out

def decode_beams(ix_to_word, beams):
    output = []
    for beam in beams:
        seq = beam['seq']
        sent = decode_sequence(ix_to_word, seq.unsqueeze(0))[0]
        output.append(sent)
    return output

def one_hot_encode(n, feat, label):
    identity = torch.eye(n)
    identity = feat.new_tensor(identity)
    return identity[label]


################################################################################
# Caption eval related util function
################################################################################
def coco_gen_format(gen_dict):
    results = []
    for k, v in gen_dict.items():
        results.append({'caption': v, 'image_id': k})
    return results

def coco_gen_format_save(gen_dict, save_path):
    results = coco_gen_format(gen_dict)
    json.dump(results, open(save_path, 'w'))

################################################################################
# Criterion related util functions
################################################################################
def to_contiguous(tensor):
    if tensor.is_contiguous():
        return tensor
    else:
        return tensor.contiguous()

class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        input = to_contiguous(input).view(-1)
        reward = to_contiguous(reward).view(-1)
        mask = (seq>0).float()
        mask = to_contiguous(torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1)).view(-1)
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, emphasize_last=False):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class EntropyLoss(nn.Module):

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, mask):
        # x: (batch, seq_length, num_weights)
        # mask: (batch, seq_length)
        batch_size = x.size(0)
        mask = mask[:, :x.size(1)]
        b = x * x.log()
        #b = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        b = b * mask.unsqueeze(2)
        b = -1.0 * b.sum() / batch_size
        return b
