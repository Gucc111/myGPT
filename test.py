import torch
from mygpt_mt import MyGPTConfig, MyGPT

src_seq = torch.tensor([[47, 48, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [13, 25, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [86, 8, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [111, 12, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [105, 6, 9, 11, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

tgt_seq = torch.tensor([[17, 0, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [87, 0, 0, 9, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

config = MyGPTConfig(src_vocab_size=184, tgt_vocab_size=201, n_layer=6, n_head=8, n_embd=512)
mygpt = MyGPT(config)

logits, loss = mygpt(src_seq, tgt_seq)

idx = torch.tensor([[47, 48, 5, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [13, 25, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
output = mygpt.generate(idx, 15)
print(output)