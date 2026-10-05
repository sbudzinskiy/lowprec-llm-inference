import os
import sys
from tqdm import tqdm

from token_provider import *
from exp_helper import *
from weight_loader import get_gpt2_files
from weight_randomizer import *
from gpt2_pretrained import load_pretrained_gpt2
from gpt2_vanilla import VanillaGPT2Model
from gpt2_lowprec import LowPrecGPT2Model, LowPrecGPT2Config

def get_token_provider(dataset_name, split, seq_len):
    try:
        dataset, text_column_name = stream_dataset_from_config(dataset_name, split=split)
        token_provider = TokenFromTextProvider(
                dataset,
                batch_size=1,
                seq_len=seq_len,
                shuffle_tokens=False,
                text_column_name=text_column_name
        )
        return token_provider
    except Exception as e:
        print(f"\nCannot load {dataset_name} ({split}): {e}")
        exit()

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
    seq_len = 1024
    model_type = sys.argv[1]
    reset_residual = int(sys.argv[2])
    reset_other = int(sys.argv[3])
    reset_ln = int(sys.argv[4])
    scale_str = sys.argv[5]
    scale = float(scale_str)
    nbatches = int(sys.argv[6])

    weights_dir = get_gpt2_files(model_type, ".")
    save_path_errors = f"./logs/logs_{model_type}_{reset_residual}{reset_other}{reset_ln}_{scale_str}x_{nbatches}bat_{seq_len}tok.csv"
    save_path_omegas = f"./logs/weights_{model_type}_{reset_residual}{reset_other}{reset_ln}_{scale_str}x.csv"

    token_provider = get_token_provider("OpenWebText", "train", seq_len)

    ref_model = load_pretrained_gpt2(model_type, weights_dir, scale).cuda()
    ref_model = reset_to_initialization(ref_model, bool(reset_residual), bool(reset_other), bool(reset_ln))

    test_model = VanillaGPT2Model(ref_model.config).cuda()
    test_model.load_state_dict(ref_model.state_dict())

    ref_model.double()
    ref_model.wte.float()
    ref_model.wpe.float()

    ref_model.eval()
    test_model.eval()

    my_bounds = {
        "omega_ov": compute_omega_ov,
        "omega_ud": compute_omega_ud
    }

    analyze_attention_bounds_across_depth(
        model=ref_model,
        functionals_dict=my_bounds,
        save_path=save_path_omegas
    )

    ref_tracker = ResidualStreamTracker(ref_model)
    comparator = ResidualStreamComparator(save_path_errors)

    with ref_tracker:
        pbar = tqdm(total=nbatches)

        for step in range(nbatches):
            pbar.set_description(f"Batch {step+1}")
            input_ids = token_provider.get_batch()
            ref_streams = ref_tracker.track_batch(step, input_ids)

            comparator.set_model(f"{scale_str}")
            comparator.register_hooks(test_model, ref_streams)
            with torch.no_grad():
                _ = test_model(input_ids)
            comparator.remove_hooks()
            pbar.update(1)
        
        pbar.close()
        token_provider.close()
        comparator.compute_and_save_global_stats()
