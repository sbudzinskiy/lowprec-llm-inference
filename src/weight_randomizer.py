import torch
import torch.nn as nn
import math

def reset_to_initialization(model, reset_residual=True, reset_other=True, reset_ln=True):
    residual_scalar = 1.0 / math.sqrt(2 * model.config.n_layer)

    with torch.no_grad():
        for name, module in model.named_modules():
            
            if isinstance(module, nn.Linear):
                d_in = module.weight.size(1)                 
                std = 1.0 / math.sqrt(d_in)

                is_residual_projection = 'c_proj' in name

                if is_residual_projection and reset_residual:
                    final_std = std * residual_scalar
                    nn.init.normal_(module.weight, mean=0.0, std=final_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif not is_residual_projection and reset_other:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias) 
            elif isinstance(module, nn.LayerNorm):
                if reset_ln:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                
    return model
