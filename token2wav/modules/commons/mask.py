import random
import torch

def random_masking(mask, mask_prob, ignore_first=True):
    assert mask.ndim == 2
    lens = mask.shape[-1]
    rand = torch.randn(mask.shape, device=mask.device)
    if ignore_first:
        rand[:, 0] = -torch.finfo(rand.dtype).max # Ignore the first item
    num_mask = min(int(lens * mask_prob), lens - 1)
    indices = rand.topk(num_mask, dim=-1).indices
    new_mask = ~torch.zeros(mask.shape, device=mask.device).scatter(1, indices, 1.).bool()
    return new_mask


def get_mask_from_lengths(lengths, max_len=None, r=1, random_mask=0.):
    if max_len is None:
        max_len = torch.max(lengths).item()
    if max_len % r != 0:
        max_len = max_len + r - max_len % r
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len).to(lengths.device))
    mask = (ids < lengths.unsqueeze(1)).bool()
    if random_mask > 0.:
        mask = mask.logical_and(random_masking(mask, random_mask))
    return mask


def compute_random_span_mask(x, mask_ratio_range=(0.5, 1.0), x_lens=None, tail_mask=False, min_span=10):
    """
    args:
        mask_ratio_range: masking ratio
        tail_mask: whether to only mask the tail
        min_span: min length of masking
    """
    batch_size, seq_len, _ = x.size()
    assert min_span <= seq_len
    if x_lens is None:
        x_lens = torch.LongTensor([seq_len]*batch_size)
    mask =  torch.full((batch_size, seq_len), False, device=x.device, dtype=torch.bool)
    for i in range(batch_size):
        mask_ratio = torch.rand(1) * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]
        mask_ratio = max(min(mask_ratio, 1 - 1e-6), 1e-6)
        mask_len = max(int(x_lens[i].item() * mask_ratio), min_span)
        if not tail_mask:
            mask_start = torch.randint(0, x_lens[i] - mask_len + 1, (1,))
        else:
            mask_start = x_lens[i] - mask_len
        mask[i, mask_start:mask_start + mask_len] = True
    return mask


def mask_data(x, mask, masking_value=0.):
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask, torch.full(x.shape, masking_value, dtype=x.dtype, device=x.device), x)

