import torch
import torch.nn as nn
import torch.distributed as dist


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, process_group=None, gather_output=False):
        super().__init__()
        self.process_group = process_group
        self.tp_size = dist.get_world_size(process_group)
        self.gather_output = gather_output
        
        out_per_rank = out_features // self.tp_size
        
        self.weight = nn.Parameter(torch.empty(in_features, out_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_per_rank))
        else:
            self.bias = None

        # initialize parameters
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x): 
        y_local = torch.matmul(x, self.weight) # (B, L, out_features/tp_size)
        if self.bias is not None:
            y_local = y_local + self.bias
        
        if self.gather_output:
            outputs = [torch.empty_like(y_local) for _ in range(self.tp_size)]
            dist.all_gather(outputs, y_local, group=self.process_group)
            y = torch.cat(outputs, dim=-1)
            return y
        else:
            return y_local


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, process_group=None, input_parallel=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.process_group = process_group
        self.tp_size = dist.get_world_size(process_group)
        self.input_parallel = input_parallel

        in_per_rank = in_features // self.tp_size

        self.weight = nn.Parameter(torch.empty(in_per_rank, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        # initialize parameters
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        if not self.input_parallel:
            x_chunks = torch.chunk(x, self.tp_size, dim=-1)
            x_local = x_chunks[dist.get_rank(self.process_group)]
        else:
            x_local = x
        
        y_local = torch.matmul(x_local, self.weight)
        dist.all_reduce(y_local, group=self.process_group, op=dist.ReduceOp.SUM)

        y = y_local
        if self.bias is not None:
            y = y + self.bias

        return y
        

class TPMLP(nn.Module):
    def __init__(self, d_model, d_ff, process_group=None):
        super().__init__()
        minimize_comm = True
        self.up_proj = ColumnParallelLinear(d_model, d_ff, bias=True, process_group=process_group, gather_output=not minimize_comm)
        self.act = nn.SiLU()
        self.down_proj = RowParallelLinear(d_ff, d_model, bias=True, process_group=process_group, input_parallel=minimize_comm)
        
    
    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    process_group = dist.group.WORLD
    torch.cuda.set_device(dist.get_rank())
    device = torch.device("cuda", dist.get_rank())
    d_modle = 512
    d_ff = 2048
    mlp = TPMLP(d_modle, d_ff, process_group=process_group).to(device)

    inputs = torch.randn(4, 4096, d_modle).to(device)
    y = mlp(inputs)
    