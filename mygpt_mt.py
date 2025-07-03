import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from typing import Optional
from model import LayerNorm, GPT

@dataclass
class MyGPTConfig:
    src_block_size: int = 512
    tgt_block_size: int = 640
    src_vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    tgt_vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # q shape: (batch_size, n_head, q_len, n_embd // n_head)
        # k shape: (batch_size, n_head, k_v_len, n_embd // n_head)
        # v shape: (batch_size, n_head, k_v_len, n_embd // n_head)
        attn_weight = torch.matmul(q / math.sqrt(self.temperature), k.transpose(-1, -2))
        # attn_weight shape: (batch_size, n_head, q_len, k_v_len)
        if mask is not None:
            attn_weight = attn_weight.masked_fill(mask==0, float('-inf'))
        
        attn_weight = self.dropout(F.softmax(attn_weight, dim=-1))
        output = torch.matmul(attn_weight, v)
        # output shape: (batch_size, n_head, q_len, n_embd)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.w_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn = ScaledDotProductAttention(config.n_embd // config.n_head, dropout=config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        # # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        # self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        # if not self.flash:
        #     print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        #     # causal mask to ensure that attention is only applied to the left in the input sequence
        #     self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
        #                                 .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, q: torch.Tensor, k_v: torch.Tensor, mask=None):
        # q/k_v shape: (batch_size, num_steps, n_embd)
        batch_size, q_len = q.shape[:2]
        k_v_len = k_v.shape[1]
        residual = q
        q = self.w_q(q)
        k = self.w_k(k_v)
        v = self.w_v(k_v)
        
        q = q.view(batch_size, q_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(batch_size, k_v_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, k_v_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        # q shape: (batch_size, n_head, q_len, n_embd)
        # k shape: (batch_size, n_head, k_v_len, n_embd)
        # v shape: (batch_size, n_head, k_v_len, n_embd)
        # # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # if self.flash:
        #     # efficient attention using Flash Attention CUDA kernels
        #     y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        q = self.attn(q, k, v, mask=mask)
        q = q.transpose(1 ,2).contiguous().view(batch_size, q_len, -1)
        # q shape: (batch_size, q_len, n_embd)
        q = self.dropout(self.proj(q))
        q += residual
        q = self.layer_norm(q)
        # q shape: (batch_size, q_len, n_embd)
        return q


class PositionwiseFeedForward(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        self.w_1    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.w_2  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = LayerNorm(config.n_embd, bias=config.bias)
    
    def forward(self, x):
        # x shape: (batch_size, q_len, n_embd)
        residual = x
        x = self.w_1(x)
        # x shape: (batch_size, q_len, 4 * n_embd)
        x = self.gelu(x)
        x = self.w_2(x)
        # x shape: (batch_size, q_len, n_embd)
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        # x shape: (batch_size, q_len, n_embd)
        return x


class Encoderlayer(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.fnn = PositionwiseFeedForward(config)
    
    def forward(self, enc_input: torch.Tensor, mask=None):
        # enc_input shape: (batch_size, src_len, n_embd)
        enc_output = self.self_attn(enc_input, enc_input, mask=mask)
        # enc_output shape: (batch_size, src_len, n_embd)
        enc_output = self.fnn(enc_output)
        # enc_output shape: (batch_size, src_len, n_embd)
        return enc_output


class Encoder(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.src_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.src_block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_stack = nn.ModuleList([Encoderlayer(config) for _ in range(config.n_layer)])
        self.config = config
    
    def forward(self, src_seq: torch.Tensor, mask=None):
        # src_seq shape: (batch_size, src_len)
        src_len = src_seq.shape[1]
        assert src_len <= self.config.src_block_size, f"Cannot forward sequence of length {src_len}, block size is only {self.config.src_block_size}"
        device = src_seq.device
        src_pos = torch.arange(0, src_len, dtype=torch.long, device=device) # shape (src_len)
        
        src_tok_emb = self.wte(src_seq)
        # src_tok_emb shape (batch_size, src_len, n_embd)
        src_pos_emb = self.wpe(src_pos) # position embeddings of shape (t, n_embd)
        # src_pos_emb shape (src_len, n_embd)
        enc_input = self.dropout(src_tok_emb + src_pos_emb)
        # enc_input shape (batch_size, src_len, n_embd)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_input, mask=mask)
            # enc_output shape (batch_size, src_len, n_embd)
        return enc_output


class Decoderlayer(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.fnn = PositionwiseFeedForward(config)
    
    def forward(self, dec_input: torch.Tensor, enc_output: torch.Tensor, self_mask=None, cross_mask=None):
        # dec_input shape: (batch_size, tgt_len, n_embd)
        # enc_output shape: (batch_size, src_len, n_embd)
        dec_output = self.self_attn(dec_input, dec_input, mask=self_mask)
        # dec_output shape: (batch_size, tgt_len, n_embd)
        dec_output = self.cross_attn(dec_output, enc_output, mask=cross_mask)
        # dec_output shape: (batch_size, tgt_len, n_embd)
        return dec_output


class Decoder(nn.Module):
    def __init__(self, config: MyGPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.tgt_vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.tgt_block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_stack = nn.ModuleList([Decoderlayer(config) for _ in range(config.n_layer)])
        self.config = config
    
    def forward(self, tgt_seq: torch.Tensor, enc_output: torch.Tensor, self_mask=None, cross_mask=None):
        # tgt_seq shape: (batch_size, tgt_len)
        tgt_len = tgt_seq.shape[1]
        assert tgt_len <= self.config.tgt_block_size, f"Cannot forward sequence of length {tgt_len}, block size is only {self.config.tgt_block_size}"
        device = tgt_seq.device
        tgt_pos = torch.arange(0, tgt_len, dtype=torch.long, device=device) # shape (src_len)
        
        tgt_tok_emb = self.wte(tgt_seq)
        # tgt_tok_emb shape (batch_size, tgt_len, n_embd)
        tgt_pos_emb = self.wpe(tgt_pos)
        # tgt_pos_emb shape (tgt_len, n_embd)
        dec_input = self.dropout(tgt_tok_emb + tgt_pos_emb)
        # dec_input shape (batch_size, tgt_len, n_embd)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_input, enc_output, self_mask=self_mask, cross_mask=cross_mask)
            # dec_output shape: (batch_size, tgt_len, n_embd)
        return dec_output


def get_pad_mask(seq: torch.Tensor, pad_idx: int):
    # seq shape: (batch_size, num_steps)
    batch_size, num_steps = seq.shape
    
    return (seq != pad_idx).view(batch_size, 1, 1, num_steps)

def get_subsequent_mask(seq: torch.Tensor):
    # seq shape: (batch_size, tgt_len)
    batch_size, tgt_len = seq.shape
    device = seq.device
    mask = torch.ones(tgt_len, tgt_len, dtype=torch.int, device=device)
    mask = (1 - torch.triu(mask, 1)).repeat(batch_size, 1, 1)
    # mask shape: (batch_size, tgt_len, tgt_len)
    return mask.unsqueeze(1) # mask shape: (batch_size, 1, tgt_len, tgt_len)


class MyGPT(GPT):
    def __init__(self, config: MyGPTConfig) -> None:
        nn.Module.__init__(self)
        assert config.src_vocab_size is not None
        assert config.tgt_vocab_size is not None
        # assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                encoder = Encoder(config),
                decoder = Decoder(config)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.tgt_vocab_size, bias=False)
        self.transformer.decoder.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
            elif pn.endswith('w_2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.encoder.wte.weight.numel()
            n_params -= self.transformer.encoder.wpe.weight.numel()
            n_params -= self.transformer.decoder.wpe.weight.numel()
        return n_params
    
    def forward(self, src_seq: torch.Tensor, tgt_seq: torch.Tensor):
        tgt_in = tgt_seq[:, :-1]
        tgt_out = tgt_seq[:, 1:].contiguous()
        src_mask = get_pad_mask(src_seq, 50259)
        tgt_mask = get_pad_mask(tgt_in, 50259) & get_subsequent_mask(tgt_in)
        
        enc_output = self.transformer.encoder(src_seq, mask=src_mask)
        dec_output = self.transformer.decoder(tgt_in, enc_output, self_mask=tgt_mask, cross_mask=src_mask)

        logits = self.lm_head(dec_output).contiguous()
        # logits shape: (batch_size, tgt_len, tgt_vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), tgt_out.view(-1), ignore_index=50259)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, src_seq: torch.Tensor, max_new_tokens: int, temperature=1, top_k=None):
        # src_seq shape: (batch_size, src_len)
        src_mask = get_pad_mask(src_seq, 50259)
        enc_output = self.transformer.encoder(src_seq, mask=src_mask)

        tgt_seq = torch.tensor([50257]).repeat(src_seq.shape[0], 1)
        # tgt_seq shape: (batch_size, 1)
        finished = torch.zeros(src_seq.shape[0], dtype=torch.bool)
        # finished shape: (batch_size,)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = tgt_seq if tgt_seq.size(1) <= self.config.tgt_block_size else tgt_seq[:, -self.config.tgt_block_size:]
            # forward the model to get the logits for the index in the sequence
            tgt_mask = get_pad_mask(idx_cond, 50259) & get_subsequent_mask(idx_cond)
            dec_output = self.transformer.decoder(idx_cond, enc_output, self_mask=tgt_mask, cross_mask=src_mask)
            logits = self.lm_head(dec_output)
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = torch.where(finished, 50259, idx_next)
            # append sampled index to the running sequence and continue
            tgt_seq = torch.cat([tgt_seq, idx_next], dim=1)
            finished |= idx_next.eq(50258)
            if finished.all():
                break
        
        return tgt_seq
    
    def crop_block_size(self, block_size):
        pass

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        pass
