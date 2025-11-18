import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import pdb

@dataclass
class Config:
    vocab_size: int = 30522
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 3
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 4096
    sliding_window: int = 128


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        return self.weight * x


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


def build_mask(seq_len, total_len, past_len=0, sliding_window=0, device="cuda"):
    mask = torch.zeros((seq_len, total_len), device=device)

    mask += torch.triu(torch.full_like(mask, float('-inf')), diagonal=past_len + 1)
    if sliding_window > 0:
        mask += torch.tril(torch.full_like(mask, float('-inf')), diagonal=past_len - sliding_window)

    return mask.view(1, 1, seq_len, total_len)


class Attention(nn.Module):
    def __init__(self, dmodel, n_heads, n_kv_heads):
        super().__init__()
        self.head_dim = dmodel // n_heads
        self.q_proj = nn.Linear(dmodel, self.head_dim * n_heads, bias=False)
        self.k_proj = nn.Linear(dmodel, self.head_dim * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(dmodel, self.head_dim * n_kv_heads, bias=False)
        self.out_proj = nn.Linear(dmodel, dmodel, bias=False)
    
    def forward(self, x, mask, past_kv, use_cache):
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, -1, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        if q.shape[1] != k.shape[1]:
            k_repeat = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v_repeat = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        # rope

        scores = torch.matmul(q, k_repeat.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores += mask
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_repeat).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if use_cache:
            present = (k, v)
        else:
            present = None
            
        return out, present


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_kv_heads):
        super().__init__()

        self.self_attn = Attention(d_model, n_heads, n_kv_heads)
        self.mlp = MLP(d_model, d_ff)
        self.input_norm = RMSNorm(d_model)
        self.post_attn_norm = RMSNorm(d_model)
    
    def forward(self, x, mask, past_kv, use_cache):
        residule = x
        x = self.input_norm(x)
        x, present = self.self_attn(x, mask, past_kv, use_cache)
        x = x + residule

        residule = x
        x = self.post_attn_norm(x)
        x = self.mlp(x)
        x = x + residule

        return x, present
        

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.n_kv_heads)
        for _ in range(config.n_layers)])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.sliding_window = config.sliding_window
        
    def forward(self, input_ids, past_kv=None, use_cache=False):

        x = self.embedding(input_ids)

        presents = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            pkv = past_kv[idx] if past_kv is not None else None
            sliding_window = self.sliding_window if idx % 2 == 0 else 0
            if past_kv is None and not use_cache:  # for training
                mask = build_mask(x.shape[1], x.shape[1], past_len=0, 
                                  sliding_window=sliding_window, device=x.device)
            else:  # for inference
                past_len = pkv[0].shape[2] if pkv is not None else 0
                mask = build_mask(x.shape[1], past_len + x.shape[1], past_len=past_len, 
                                  sliding_window=sliding_window, device=x.device)
            x, present = layer(x, mask, pkv, use_cache)

            if use_cache:
                presents.append(present)

        return x, presents
    
    @torch.no_grad()
    def generate(self, 
                 input_ids,
                 temperature=1.0,
                 top_k=None,
                 top_p=None,
                 max_new_tokens=50,
                 eos_token_id=None):
        self.eval()
        device = input_ids.device

        # Prefill and build cache
        hidden, past = self.forward(input_ids, past_kv=None, use_cache=True)
        logits = self.lm_head(hidden)
        next_token_logits = logits[:, -1, :]
        
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
        generated = input_ids

        for _ in range(max_new_tokens):
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k is not None:
                k = min(top_k, next_token_logits.shape[-1])
                values, indices = torch.topk(next_token_logits, k)
                invalid_indices = next_token_logits < values[:, -1][:, None]
                next_token_logits[invalid_indices] = float('-inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cum_prob = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                invalid_indices = cum_prob > top_p
                # keep the first token above the threshold
                invalid_indices[..., 1:] = invalid_indices[..., :-1].clone()
                # first token is always valid
                invalid_indices[..., 0] = 0

                remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                remove = remove.scatter(dim=1, index=sorted_indices, src=invalid_indices)
                
                next_token_logits[remove] = float('-inf')


            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None:
                newly_finished = next_token.squeeze(-1) == eos_token_id
                finished = finished | newly_finished
                if finished.all():
                    break

            input_ids = next_token

            hidden, past = self.forward(input_ids, past_kv=past, use_cache=True)
            logits = self.lm_head(hidden)
            next_token_logits = logits[:, -1, :]

        return generated


if __name__ == "__main__":
    config = Config()
    model = Transformer(config)
    model = model.to("cuda")

    batch_size = 8
    sample_input = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len)).to("cuda")
    print(f"Sample input shape: {sample_input.shape}")

    # training
    hidden, _ = model(sample_input)
    logits = model.lm_head(hidden)
    print(f"Logits shape (training): {logits.shape}")
    loss = F.cross_entropy(
        logits[:, :-1, :].contiguous().view(-1, config.vocab_size),
        sample_input[:, 1:].contiguous().view(-1)
    )
    print(f"Loss: {loss.item()}")

    # inference
    out = model.generate(sample_input, top_p=0.95, max_new_tokens=20, eos_token_id=50256)
    print(f"Generated shape (inference): {out.shape}")
