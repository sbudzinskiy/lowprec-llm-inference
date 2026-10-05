import os
import torch
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------------------

class ModelSuite:
    def __iter__(self):
        raise NotImplementedError("Subclasses must implement __iter__ yielding (model_name, model)")

class ConcatenatedModelSuite:
    def __init__(self, suites):
        self.suites = suites

    def __iter__(self):
        for suite in self.suites:
            for model_name, model in suite:
                yield model_name, model

class BasicModelSuite(ModelSuite):
    def __init__(self, models_list):
        self.models_list = models_list

    def __iter__(self):
        for model_name, model in self.models_list:
            yield model_name, model

class LowPrecGPT2ModelSuite(ModelSuite):
    def __init__(self, base_model, variations):
        self.model = base_model
        self.variations = variations

    def _set_variation(self, target_gemm, m_bits):
        reset_model_precision(self.model)
        update_model_precision(self.model, target_gemm, m_bits)

    def __iter__(self):
        for model_name, target_gemm, m_bits in self.variations:
            self._set_variation(target_gemm, m_bits)
            yield model_name, self.model

# ---------------------------------------------------------------------------------------------------------------------------

def update_model_precision(model, target_gemm, m_bits):
    setattr(model.config, target_gemm, m_bits)

    for block in model.h:
        if target_gemm == 'm_bits_mlp_fc': block.mlp.c_fc.m_bits = m_bits
        elif target_gemm == 'm_bits_mlp_act': block.mlp.act.m_bits = m_bits
        elif target_gemm == 'm_bits_mlp_proj': block.mlp.c_proj.m_bits = m_bits
        elif target_gemm == 'm_bits_attn_qkv': block.attn.c_attn.m_bits = m_bits
        elif target_gemm == 'm_bits_attn_score': block.attn.m_bits_score = m_bits
        elif target_gemm == 'm_bits_attn_softmax': block.attn.m_bits_softmax = m_bits
        elif target_gemm == 'm_bits_attn_value': block.attn.m_bits_value = m_bits
        elif target_gemm == 'm_bits_attn_proj': block.attn.c_proj.m_bits = m_bits
            
    if target_gemm == 'm_bits_lm_proj':
        model.lm_head.m_bits = m_bits

def reset_model_precision(model, default_bits=23):
    targets = [
        'm_bits_mlp_fc', 'm_bits_mlp_act', 'm_bits_mlp_proj', 
        'm_bits_attn_qkv', 'm_bits_attn_score', 'm_bits_attn_softmax', 'm_bits_attn_value', 'm_bits_attn_proj',
        'm_bits_lm_proj'
    ]
    for t in targets:
        update_model_precision(model, t, default_bits)

# ---------------------------------------------------------------------------------------------------------------------------

class ResidualStreamTracker:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.num_streams = len(self.model.h) * 2
        self.ram_cache = [None] * self.num_streams
        self.batch_hash = None

    def track_batch(self, batch_hash, input_ids):
        if self.batch_hash != batch_hash:
            self._clear_cache()
            
            with torch.no_grad():
                _ = self.model(input_ids)
                
            self.batch_hash = batch_hash
            
        return self.ram_cache

    def _get_pre_hook(self, idx):
        def hook(module, inputs):
            self.ram_cache[idx] = inputs[0].detach().cpu()
        return hook

    def _get_post_hook(self, idx):
        def hook(module, inputs, output):
            stream = output[0] if isinstance(output, tuple) else output
            self.ram_cache[idx] = stream.detach().cpu()
        return hook

    def register_hooks(self):
        self.remove_hooks()
        for i, block in enumerate(self.model.h):
            h1 = block.ln_2.register_forward_pre_hook(self._get_pre_hook(2 * i))
            h2 = block.register_forward_hook(self._get_post_hook(2 * i + 1))
            self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        
    def _clear_cache(self):
        self.ram_cache = [None] * self.num_streams

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

# ---------------------------------------------------------------------------------------------------------------------------

class ResidualStreamComparator:
    def __init__(self, save_path):
        self.save_path = save_path        
        self.hooks = []
        self.active_model_name = None
        self.history = {}

    def set_model(self, model_name):
        self.active_model_name = model_name
        if model_name not in self.history:
            self.history[model_name] = {}

    def _get_pre_hook(self, idx, ref_streams):
        def hook(module, inputs):
            self._process_stream(idx, inputs[0].detach(), ref_streams[idx])
        return hook

    def _get_post_hook(self, idx, ref_streams):
        def hook(module, inputs, output):
            stream = output[0] if isinstance(output, tuple) else output
            self._process_stream(idx, stream.detach(), ref_streams[idx])
        return hook
        
    def _process_stream(self, idx, test_stream, ref_stream):
        ref_stream = ref_stream.to(test_stream.device)
        
        rel_comp, rel_linf, abs_linf = compute_errors(test_stream, ref_stream)
        min_l2 = compute_min_token_l2_norm(ref_stream)

        layer_dict = self.history[self.active_model_name]
        if idx not in layer_dict:
            layer_dict[idx] = {"min_token_l2": [], "rel_comp": [], "rel_linf": [], "abs_linf": []}
            
        layer_dict[idx]["min_token_l2"].append(min_l2.cpu())
        layer_dict[idx]["rel_comp"].append(rel_comp.cpu())
        layer_dict[idx]["rel_linf"].append(rel_linf.cpu())
        layer_dict[idx]["abs_linf"].append(abs_linf.cpu())

    def register_hooks(self, active_model, ref_streams):
        self.remove_hooks()
        for i, block in enumerate(active_model.h):
            h1 = block.ln_2.register_forward_pre_hook(self._get_pre_hook(2 * i, ref_streams))
            h2 = block.register_forward_hook(self._get_post_hook(2 * i + 1, ref_streams))
            self.hooks.extend([h1, h2])

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def compute_and_save_global_stats(self):
        print("Computing statistics for test models...")
        rows = []

        for model_name, layer_dict in self.history.items():
            for layer_idx in sorted(layer_dict.keys()):
                metrics = layer_dict[layer_idx]
                
                global_stats = {
                    "min_token_l2": compute_tensor_stats(torch.cat(metrics["min_token_l2"])),
                    "rel_comp": compute_tensor_stats(torch.cat(metrics["rel_comp"])),
                    "rel_linf": compute_tensor_stats(torch.cat(metrics["rel_linf"])),
                    "abs_linf": compute_tensor_stats(torch.cat(metrics["abs_linf"]))
                }
                
                row_data = {"model": model_name, "layer": layer_idx}
                for metric_name, stats in global_stats.items():
                    for stat_name, val in stats.items():
                        row_data[f"{metric_name}_{stat_name}"] = val
                        
                rows.append(row_data)
               
        df = pd.DataFrame(rows)
        df.to_csv(self.save_path, index=False, float_format='%.4e')
        print(f"Successfully saved global statistics to {self.save_path}")

# ---------------------------------------------------------------------------------------------------------------------------

def compute_min_token_l2_norm(ref_stream):
    ref = ref_stream.double()
    token_norms = torch.linalg.norm(ref, ord=2, dim=-1)    
    min_norm = token_norms.min(dim=1).values
    
    return min_norm

def compute_errors(test_stream, ref_stream):
    ref = ref_stream.double()
    diff = test_stream - ref
    
    # Componentwise
    denom_comp = ref.abs()
    rel_comp = torch.where(denom_comp != 0, diff.abs() / denom_comp, 0.0)
    rel_comp = rel_comp.max(dim=-1).values.flatten()
    
    # L_inf
    diff_linf = diff.abs().max(dim=-1).values
    ref_linf = ref.abs().max(dim=-1).values
    rel_linf = torch.where(ref_linf != 0, diff_linf / ref_linf, 0.0).flatten()
    abs_linf = diff_linf.flatten()
        
    return rel_comp, rel_linf, abs_linf

def compute_tensor_stats(tensor):
    q_vals = torch.tensor(
        [0.01, 0.25, 0.50, 0.75, 0.99], 
        device=tensor.device, 
        dtype=torch.float64
    )
    quantiles = torch.quantile(tensor, q_vals)
    
    return {
        "p01": quantiles[0].item(),
        "p25": quantiles[1].item(),
        "p50": quantiles[2].item(),
        "p75": quantiles[3].item(),
        "p99": quantiles[4].item()
    }
