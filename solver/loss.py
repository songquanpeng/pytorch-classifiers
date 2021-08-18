import torch
import torch.nn.functional as F
from munch import Munch


def compute_loss(nets, args, sample):
    loss = F.nll_loss(sample.x, sample.y)

    loss = loss_real + loss_fake
    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item())


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
