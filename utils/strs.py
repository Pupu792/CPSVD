import os
import sys
import torch
import torch.nn as nn

def binary_search_truncation_rank(model, args):
    sensitivity_dict = torch.load(args.sensitivity_path)
    print(sensitivity_dict)
    if 'lm_head' in sensitivity_dict:
        del sensitivity_dict['lm_head']
    module_dict = {name: module for name, module in model.named_modules()}
    print(module_dict)


    ratio_target = args.ratio
    default_param_ratio = 1


    sensitivity_list = []
    for layername, v in sensitivity_dict.items():
        for param_ratio, ppl in v.items():
            if param_ratio >= 1:
                # we need to compress the weights, so parameter ratio should be less than 1
                continue
            sensitivity_list.append((layername, param_ratio, ppl))
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])

    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    assert ratio_target > 0

    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
        for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
        tot_params = 0
        compress_params = 0

        for layername, param_ratio in layers_min_ratio.items():
            raw_linear = module_dict[layername]
            tot_params += raw_linear.weight.numel()
            compress_params += raw_linear.weight.numel() * param_ratio
        now_ratio = compress_params / tot_params

        msg = f"low={low} mid={mid}, high={high}, now_ratio={now_ratio}, params=({compress_params}/{tot_params})"
        print(msg)
        if now_ratio > ratio_target:
            high = mid
        else:
            low = mid + 1

    print(f"=== Searching done, decomposing layers... ===")
    layers_min_ratio = {layername: default_param_ratio for layername in sensitivity_dict.keys()}
    for layername, param_ratio, ppl in sorted_sensitive_list[mid:]:
        if layers_min_ratio[layername] is None:
            layers_min_ratio[layername] = param_ratio
        else:
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)

    del model 
    torch.cuda.empty_cache()
    return layers_min_ratio