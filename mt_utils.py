import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pickle
import math
from collections import Counter
from typing import Tuple, List


data_dir = os.path.join('data', 'EuroPat')
meta_path = os.path.join(data_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
BOS_ID = meta['<bos>']
EOS_ID = meta['<eos>']
PAD_ID = meta['<pad>']


class MTDataset(Dataset):
    def __init__(
        self,
        src_bin: str,
        src_off: str,
        tgt_bin: str,
        tgt_off: str
    ):
        # 加载 memmap 和 offsets
        self.src = np.memmap(src_bin, dtype=np.uint16, mode='r')
        self.tgt = np.memmap(tgt_bin, dtype=np.uint16, mode='r')
        self.src_offsets = np.load(src_off)
        self.tgt_offsets = np.load(tgt_off)
        assert len(self.src_offsets) == len(self.tgt_offsets), "源/目标句子数不一致"
        # 有效句对数
        self.n_sent = len(self.src_offsets) - 1

    def __len__(self) -> int:
        return self.n_sent

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 切出第 idx 句的 token IDs
        s_start, s_end = self.src_offsets[idx], self.src_offsets[idx+1]
        t_start, t_end = self.tgt_offsets[idx], self.tgt_offsets[idx+1]
        src_ids = torch.from_numpy(self.src[s_start:s_end].astype(np.int64))
        tgt_ids = torch.from_numpy(self.tgt[t_start:t_end].astype(np.int64))
        return src_ids, tgt_ids


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]], bos_id: int = BOS_ID, eos_id: int = EOS_ID, pad_id: int = PAD_ID) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据 batch 中各句长度动态 padding
    """
    src_seqs, tgt_seqs = zip(*batch)
    # 拼接 <bos> 和 <eos>
    tgt_seqs_list = []
    for tgt_seq in tgt_seqs:
        tgt_seqs_list.append(torch.concat([torch.tensor([bos_id]), tgt_seq, torch.tensor([eos_id])]))
    # 计算最大长度
    max_src = max(seq.size(0) for seq in src_seqs)
    max_tgt = max(seq.size(0) for seq in tgt_seqs_list)
    B = len(batch)
    # 构造批次矩阵
    src_batch = torch.full((B, max_src), pad_id, dtype=torch.long)
    tgt_batch = torch.full((B, max_tgt), pad_id, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs_list)):
        src_batch[i, :s.size(0)] = s
        tgt_batch[i, :t.size(0)] = t
    return src_batch, tgt_batch


def build_dataloader(
    src_bin: str,
    src_off: str,
    tgt_bin: str,
    tgt_off: str,
    batch_size: int,
    num_workers: int = 4
) -> DataLoader:
    """
    构建 MT DataLoader：
    - 每 epoch 无放回随机遍历所有句对
    - 动态 padding via collate_fn
    """
    dataset = MTDataset(src_bin, src_off, tgt_bin, tgt_off)
    sampler = RandomSampler(dataset, replacement=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def compute_bleu(
    preds: torch.Tensor,
    tgt_seq: torch.Tensor,
    pad_id: int = 50259,
    eos_id: int = 50258,
    max_n: int = 4
) -> float:
    """
    计算一个 batch 上的 corpus-level BLEU 分数。

    参数:
        preds:   LongTensor, shape (B, pred_len)，模型输出 token IDs
        tgt_seq: LongTensor, shape (B, tgt_len)，参考序列 token IDs
        pad_id:  int, padding token 的 ID
        eos_id:  int, EOS token 的 ID
        max_n:   int, 最大 n-gram 阶数（默认为 4，即 BLEU-4）
    返回:
        bleu:    float, corpus-level BLEU 分数
    """
    # 转成 Python list 方便处理
    if isinstance(preds, torch.Tensor):
        preds = preds.tolist()
    if isinstance(tgt_seq, torch.Tensor):
        tgt_seq = tgt_seq.tolist()

    # 累积 n-gram 匹配数和总数
    total_clipped = [0] * max_n
    total_counts  = [0] * max_n
    total_ref_len = 0
    total_hyp_len = 0

    for hyp_ids, ref_ids in zip(preds, tgt_seq):
        # 去 EOS（及之后）并过滤掉 PAD
        if eos_id in hyp_ids:
            hyp_ids = hyp_ids[: hyp_ids.index(eos_id)]
        hyp_tokens = [t for t in hyp_ids if t != pad_id]

        ref_ids = ref_ids[: ref_ids.index(eos_id)]
        ref_tokens = [t for t in ref_ids if t != pad_id]

        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        # 各级 n-gram 统计
        for i in range(max_n):
            n = i + 1
            hyp_ngrams = Counter(
                tuple(hyp_tokens[j:j+n]) 
                for j in range(len(hyp_tokens)-n+1)
            )
            ref_ngrams = Counter(
                tuple(ref_tokens[j:j+n]) 
                for j in range(len(ref_tokens)-n+1)
            )
            clipped = sum(
                min(count, ref_ngrams.get(ng, 0)) 
                for ng, count in hyp_ngrams.items()
            )
            total_clipped[i] += clipped
            total_counts[i]  += sum(hyp_ngrams.values())

    # 1–n-gram 精确度
    precisions = [
        (total_clipped[i] / total_counts[i]) if total_counts[i] > 0 else 0.0
        for i in range(max_n)
    ]

    # 几何平均
    if min(precisions) == 0.0:
        geo_mean = 0.0
    else:
        log_sum = sum((1.0/max_n) * math.log(p) for p in precisions)
        geo_mean = math.exp(log_sum)

    # brevity penalty
    if total_hyp_len == 0:
        bp = 0.0
    elif total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - total_ref_len/total_hyp_len)

    return bp * geo_mean
