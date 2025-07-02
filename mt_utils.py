import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import pickle
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
