import os
import sys
import torch
import math
import pandas as pd
from tqdm import tqdm

from weight_loader import get_gpt2_files
from gpt2_pretrained import load_pretrained_gpt2
from gpt2_vanilla import VanillaGPT2Model

def analyze_attention_bounds_across_depth(model, functionals_dict, save_path):
    rows = []
    
    for depth_idx, block in enumerate(model.h):
        row_data = {"depth": depth_idx}
        
        for func_name, func_callable in functionals_dict.items():
            row_data[func_name] = func_callable(block, model.config)
            
        rows.append(row_data)
                
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, float_format='%.4e')
    print(f"Successfully saved block analysis to {save_path}")

def compute_omega_ov(block, config):
    n_head = config.n_head
    d_model = config.n_embd
    d_head = d_model // n_head

    W_O = block.attn.c_proj.weight.detach().double()
    W_QKV = block.attn.c_attn.weight.detach().double()
    
    norm_W_O = torch.linalg.norm(W_O, ord=torch.inf)
    _, _, W_V = W_QKV.chunk(3, dim=0)

    max_V_norm = 0.0
    for h in range(n_head):
        W_V_h = W_V[h * d_head : (h + 1) * d_head, :]
        
        norm_h = torch.linalg.norm(W_V_h, ord=torch.inf)
        if norm_h > max_V_norm:
            max_V_norm = norm_h

    return (norm_W_O * max_V_norm).item()

def compute_omega_ud(block, config):
    W_up = block.mlp.c_fc.weight.detach().double()
    W_down = block.mlp.c_proj.weight.detach().double()
    
    norm_up = torch.linalg.norm(W_up, ord=torch.inf)
    norm_down = torch.linalg.norm(W_down, ord=torch.inf)
    
    return (norm_up * norm_down).item()

def compute_omega_kq(block, config):
    n_head = config.n_head
    d_model = config.n_embd
    d_head = d_model // n_head

    W_QKV = block.attn.c_attn.weight.detach().double()
    W_Q, W_K, _ = W_QKV.chunk(3, dim=0)

    max_product = 0.0
    for h in range(n_head):
        W_Q_h = W_Q[h * d_head : (h + 1) * d_head, :]
        W_K_h = W_K[h * d_head : (h + 1) * d_head, :]

        k_row_sums = W_K_h.abs().sum(dim=1)
        k_norm = torch.linalg.norm(k_row_sums, ord=2)

        q_row_sums = W_Q_h.abs().sum(dim=1)
        q_norm = torch.linalg.norm(q_row_sums, ord=2)

        product = k_norm * q_norm
        if product > max_product:
            max_product = product

    scalar = 2.0 / math.sqrt(d_head)
    
    return (scalar * max_product).item()

if __name__ == "__main__":
    model_type = sys.argv[1]
    weights_dir = get_gpt2_files(model_type, ".")
    save_path = f"./logs/weights_{model_type}.csv"
    
    model = load_pretrained_gpt2(model_type, weights_dir).cuda().eval()

    my_bounds = {
        "omega_ov": compute_omega_ov,
        "omega_ud": compute_omega_ud
    }

    analyze_attention_bounds_across_depth(
        model=model,
        functionals_dict=my_bounds,
        save_path=save_path
    )
