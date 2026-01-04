#coding:utf8
import os
import sys
import argparse
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
from accelerate import dispatch_model
import json
import torch.nn.functional as F
import math
import gc

from utils.data_utils import *
from utils.model_utils import *
from evaluater import * 
from component.cal_r import seg_W_SVD, analyze_column_loss
from component.cal_r_low_rank import seg_W_SVD_v2
from component.cpsvd_linear import *

from utils.svd_utils import hessian_weight_svd, calib_sensitivity_ppl, calib_sensitivity_div, calib_sensitivity_kl, calib_sensitivity_ppl_asvd, calib_sensitivity_ppl_dynamic
from utils.rank_search import determine_optimal_ranks_dp, determine_optimal_ranks_greedy, apply_importance_weighting

from utils.post_process import *
from utils.whiten_dynamic import *

import lm_eval
from lm_eval.models.huggingface import HFLM

import sys
tqdm.write = lambda msg: sys.stderr.write(msg + '\n')

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev):
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        # for name, module in model.named_modules():
            # if 'rotary_emb' in name:
            #     module.inv_freq = module.inv_freq.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None,
             "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs.get('position_ids', None)
                cache['position_embeddings'] = kwargs.get(
                    'position_embeddings', None)
            else:
                cache['attention_mask'] = torch.cat(
                    (cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if kwargs.get('position_ids', None) is not None:
                    cache['position_ids'] = torch.cat(
                        (cache['position_ids'], kwargs['position_ids']), dim=0)
                # if kwargs.get('position_embeddings', None) is not None:
                #     cache['position_embeddings'] = torch.cat((cache['position_embeddings'], kwargs['position_embeddings']), dim=0)
            raise ValueError
            
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)

    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    hessian_mats = {}
    for i in tqdm(range(len(layers))):
        layer_hessian = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if position_ids is None:
                outs[j] = layer(inps[j].unsqueeze(
                    0), attention_mask=attention_masks[j].unsqueeze(0))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(
                    0), position_embeddings=position_embeddings)[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            layer_hessian[name] = subset[name].scaling_diag_matrix.cpu()
            subset[name].scaling_diag_matrix = None
            del subset[name].scaling_diag_matrix
            # subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()

        hessian_mats[i] = layer_hessian
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()
    return hessian_mats
    # return hessian_mat

@torch.no_grad()
def whitening(model_name, model, hessian_mat, ratio, dev, args):
    """
    使用 CPSVD_Linear 类对模型进行白化和SVD分解，并进行原位替换。
    """
    model.eval()
    
    layers = model.model.layers
    layer_prefix = 'model.layers'

    model_id = model.config._name_or_path.split('/')[-1].replace('/', '_')
    ratio_str = f"{ratio:.2f}".replace('.', '_') # e.g., 0.70 -> 0_70
    cache_file_ranks = f"cache/{model_id}_optimal_ranks_{args.strategy}_ratio{ratio_str}.pt"


    if args.strategy == 'uniform':
        pass
    elif args.strategy == 'strs':
        from utils.strs import binary_search_truncation_rank
        strs_ratios = binary_search_truncation_rank(model, args)
    elif os.path.exists(cache_file_ranks):
        print(f"Loading cached optimal ranks from: {cache_file_ranks}")
        optimal_ranks = torch.load(cache_file_ranks, map_location="cpu")
    # --- 3. Compute if cache doesn't exist ---
    else:
        sensitivity_dict = torch.load(args.sensitivity_path)
        print(f"Cache file not found. Computing optimal ranks...")
        if args.strategy == 'greedy':
            optimal_ranks = determine_optimal_ranks_greedy(model, sensitivity_dict, ratio, args)
        elif args.strategy == 'dp':
            optimal_ranks = determine_optimal_ranks_dp(model, sensitivity_dict, ratio, args)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        # --- 4. Save the computed result ---
        print(f"Computation finished. Saving optimal ranks to: {cache_file_ranks}")
        os.makedirs(os.path.dirname(cache_file_ranks), exist_ok=True) # Ensure cache directory exists
        torch.save(optimal_ranks, cache_file_ranks)
    
    print("开始使用 CPSVD_Linear 进行白化和SVD分解替换...")
    for i in tqdm(range(len(layers)), desc="处理模型层"):
        layer = layers[i]
        # find_layers 应该返回一个字典，如 {'q_proj': q_proj_layer, ...}
        subset = find_layers(layer)

        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"

            original_layer = subset[name]
            W = original_layer.weight.data.float().to(dev)

            if args.strategy == 'uniform':
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            elif args.strategy == 'strs':
                strs_name = 'model.layers.' + str(i) + '.' + name
                ratio = strs_ratios[strs_name]
                if ratio == 1:
                    continue
                else:
                    num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            else:
                if full_name not in optimal_ranks or optimal_ranks[full_name] is None:
                    print(f"模块 {full_name} 不进行压缩，跳过。")
                    continue
                
                num_s_after_trunc = optimal_ranks[full_name]
            
            # --- SVD计算部分 (与您原始代码相同) ---
            raw_scaling_diag_matrix = hessian_mat[i][name]
            
            scale_float = True if 'Qwen3' in args.model else False
            U, S, VT, scaling_matrix_inv = hessian_weight_svd(W, raw_scaling_diag_matrix, dev, scale_float)

            
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv.float())
            
            sqrt_sigma = torch.diag(torch.sqrt(truc_s))
            
            # svd_u: [out_features, rank], svd_v: [rank, in_features]
            svd_u = torch.matmul(truc_u, sqrt_sigma)
            svd_v = torch.matmul(sqrt_sigma, truc_v)

            svd_u = svd_u.cpu().to(original_layer.weight.dtype)
            svd_v = svd_v.cpu().to(original_layer.weight.dtype)

                # svd_u_v2, svd_v_v2, _ = hessian_weight_svd_v2(W, raw_scaling_diag_matrix, double=True)
                # svd_u_v2 = svd_u_v2[:,:num_s_after_trunc]
                # svd_v_v2 = svd_v_v2[:num_s_after_trunc,:]
                # w_lr_v2 = svd_u_v2@svd_v_v2
                # error = torch.trace(w_lr_v2@H@w_lr_v2.T)
                # print(error)
            # 1. 创建 CPSVD_Linear 实例。
            #    关键：传入 col_indices=[] 使其工作在纯SVD模式。
            new_svd_layer = CPSVD_Linear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=num_s_after_trunc,
                col_indices=[],  # <--- 让 CPSVD_Linear 工作在纯SVD模式
                bias=original_layer.bias is not None
            )

            # 2. 将计算出的低秩矩阵和偏置赋给新层。
            #    v_proj 的权重维度是 [rank, in_features]
            #    u_proj 的权重维度是 [out_features, rank]
            new_svd_layer.v_proj.weight.data = svd_v
            new_svd_layer.u_proj.weight.data = svd_u
            if original_layer.bias is not None:
                new_svd_layer.u_proj.bias.data = original_layer.bias.data

            # 3. 获取父模块，并将原始层替换为新创建的SVD层。
            full_layer_name = f"{layer_prefix}.{i}.{name}"
            parent_module, module_name = get_parent_module_and_name(model, full_layer_name)
            setattr(parent_module, module_name, new_svd_layer)
            

            del W, U, S, VT, truc_s, truc_u, truc_v, sqrt_sigma, svd_u, svd_v

            torch.cuda.empty_cache()
            gc.collect()
    print("模型所有指定层已成功替换为 CPSVD_Linear 层。")

@torch.no_grad()
def calculate_error_traverse(model_name, model, hessian_mat, ratio, dev, args):
    model.eval()

    layers = model.model.layers
    layer_prefix = 'model.layers'

    layer_subset = [0]
    for i in tqdm(layer_subset, desc="处理模型层"):
        layer = layers[i]
        subset = find_layers(layer)

        layer_errors = {}
        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"

            original_layer = subset[name]
            
            # 确保原始层在目标设备上进行计算
            W = original_layer.weight.data.float().to(dev) 
            
            num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            # --- Step 4: SVD计算 ---
            # 确保 Hessian 片段也在目标设备上
            # --- 解析正确的 hessian_mat name ---
            hessian_parts = full_name.split('.')
            hessian_i = int(hessian_parts[2])
            hessian_name = ".".join(hessian_parts[3:])
            if hessian_i not in hessian_mat or hessian_name not in hessian_mat[hessian_i]:
                 print(f"Warning: Hessian info not found for layer {hessian_i}, name '{hessian_name}'. Skipping compression for {full_name}.")
                 continue
            raw_scaling_diag_matrix = hessian_mat[hessian_i][hessian_name].to(dev)
            # --- 解析结束 ---

            scale_float = 'Qwen3' in args.model
            
            # 调用 seg_W_SVD
            weights_errors = analyze_column_loss(W, raw_scaling_diag_matrix, num_s_after_trunc, scale_float, step=10)
            layer_errors[name] = weights_errors
        
        torch.save(layer_errors, f"layer_errors/{model_name.split('/')[-1]}_{i}_{args.ratio}.pt")
    




@torch.no_grad()
def whitening_cp(model_name, model, hessian_mat, ratio, dev, args):
    """
    使用 CPSVD_Linear 类对模型进行白化和SVD分解，并进行原位替换。
    [修改]：修正了内存清理，并添加了保存 error metrics 的功能。
    """
    model.eval()
    
    # 确定模型层的路径前缀
    if 'opt' in model_name:
        layers = model.model.decoder.layers
        layer_prefix = 'model.decoder.layers'
    else:
        layers = model.model.layers
        layer_prefix = 'model.layers'

    print(f"Determining optimal ranks using strategy: {args.strategy} for target ratio {ratio:.2f}")

    # --- 1. Define cache file path ---
    # Include strategy and ratio in the filename for uniqueness
    # Assuming args has attributes like model name/id, dataset, seed etc. used for sensitivity cache
    model_id = model.config._name_or_path.split('/')[-1].replace('/', '_')
    ratio_str = f"{ratio:.2f}".replace('.', '_') # e.g., 0.70 -> 0_70
    cache_file_ranks = f"cache/{model_id}_optimal_ranks_{args.strategy}_ratio{ratio_str}.pt"

    # --- 2. Check if cache exists ---
    if args.strategy == 'uniform':
        pass
    # elif os.path.exists(cache_file_ranks):
    #     print(f"Loading cached optimal ranks from: {cache_file_ranks}")
    #     optimal_ranks = torch.load(cache_file_ranks, map_location="cpu")

    # --- 3. Compute if cache doesn't exist ---
    else:
        if args.strategy == 'greedy':
            sensitivity_dict = torch.load(args.sensitivity_path)
            optimal_ranks = determine_optimal_ranks_greedy(model, sensitivity_dict, ratio, args)
        elif args.strategy == 'dp':
            sensitivity_dict = torch.load(args.sensitivity_path)
            optimal_ranks = determine_optimal_ranks_dp(model, sensitivity_dict, ratio, args)
        elif args.strategy == 'bolaco':
            if 'Llama-2-7b' not in model_name:
                raise ValueError('Bolaco only supports Llama-2-7b model')
            if args.ratio == 0.8:
                optimal_ranks = {'q_proj':744, 'k_proj':744, 'o_proj':1616, 'gate_proj':2512, 'up_proj':2408}
            elif args.ratio == 0.7:
                optimal_ranks = {'q_proj':656, 'k_proj':656, 'o_proj':1392, 'gate_proj':2128, 'up_proj':2352, 'down_proj': 2312}
            else:
                raise ValueError('bolaco only support param ratio 0.8 and 0.7')
        elif args.strategy == 'sola':
            if 'Llama-2-7b' not in model_name:
                raise ValueError('SoLA only supports Llama-2-7b model')
            if args.ratio == 0.8:
                optimal_ranks = {"q_proj": 1120, "k_proj": 800, "gate_proj": 1760, "up_proj": 2400, "down_proj": 2240}
            elif args.ratio == 0.7:
                optimal_ranks = {"q_proj": 640, "k_proj": 640, "o_proj": 1440, "gate_proj": 1760, "up_proj": 1920, "down_proj": 1760}
            else:
                raise ValueError('SoLA only support param ratio 0.8 and 0.7')
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

        # --- 4. Save the computed result ---
        print(f"Computation finished. Saving optimal ranks to: {cache_file_ranks}")
        os.makedirs(os.path.dirname(cache_file_ranks), exist_ok=True) # Ensure cache directory exists
        torch.save(optimal_ranks, cache_file_ranks)

    # --- optimal_ranks is now ready to use ---
    print("Optimal ranks determined.")

    # --- Step 2: 初始化 error_metrics 字典 ---
    error_metrics = {}

    print("开始使用 CPSVD_Linear 进行白化和SVD分解替换...")
    for i in tqdm(range(len(layers)), desc="处理模型层"):
        layer = layers[i]
        subset = find_layers(layer)
        
        # --- 为当前层初始化 error_metrics 条目 ---
        error_metrics[f"layer_{i}"] = {}

        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"

            original_layer = subset[name]
            
            # 确保原始层在目标设备上进行计算
            W = original_layer.weight.data.float().to(dev) 
            
            if args.strategy == 'uniform':
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            elif args.strategy == 'bolaco' or args.strategy == 'sola':
                proj_name = name.split('.')[1]
                if proj_name in optimal_ranks:
                    num_s_after_trunc = optimal_ranks[proj_name]
                else:
                    continue
            else:
                if full_name not in optimal_ranks or optimal_ranks[full_name] is None:
                    print(f"模块 {full_name} 不进行压缩，跳过。")
                    continue
                num_s_after_trunc = optimal_ranks[full_name]
            # --- Step 4: SVD计算 ---
            # 确保 Hessian 片段也在目标设备上
            # --- 解析正确的 hessian_mat name ---
            hessian_parts = full_name.split('.')
            hessian_i = int(hessian_parts[2])
            hessian_name = ".".join(hessian_parts[3:])
            if hessian_i not in hessian_mat or hessian_name not in hessian_mat[hessian_i]:
                 print(f"Warning: Hessian info not found for layer {hessian_i}, name '{hessian_name}'. Skipping compression for {full_name}.")
                 continue
            raw_scaling_diag_matrix = hessian_mat[hessian_i][hessian_name].to(dev)
            # --- 解析结束 ---

            scale_float = 'Qwen3' in args.model
            
            # 调用 seg_W_SVD
            truc_wu, truc_wv, col_w, svd_rank, keep_indices, drop_indices, sigma_error_reduction, svd_error_reduction = seg_W_SVD(W, raw_scaling_diag_matrix, num_s_after_trunc, scale_float)

            # --- Step 5: 保存 Error Metrics ---
            # 将 .item() 用于标量张量，确保可序列化
            error_metrics[f"layer_{i}"][name] = {
                "sigma_error_reduction": sigma_error_reduction.item() if torch.is_tensor(sigma_error_reduction) else sigma_error_reduction,
                "svd_error_reduction": svd_error_reduction.item() if torch.is_tensor(svd_error_reduction) else svd_error_reduction
            }
            print(error_metrics)
            # --- Step 6: 准备替换层 ---
            # 将计算得到的权重移回CPU并转换类型 (如果CPSVD_Linear期望如此)
            # 注意：如果CPSVD_Linear设计为在GPU上工作，这里的 .cpu() 可能需要移除
            truc_wu_cpu = truc_wu.cpu().to(original_layer.weight.dtype)
            truc_wv_cpu = truc_wv.cpu().to(original_layer.weight.dtype)
            col_w_cpu = col_w.cpu().to(original_layer.weight.dtype)
            
            # --- Step 7: 使用 CPSVD_Linear 进行原位替换 ---
            new_svd_layer = CPSVD_Linear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=svd_rank, # rank 应该是实际使用的奇异值数量
                col_indices=keep_indices, 
                bias=original_layer.bias is not None
            )

            # --- 修正权重赋值 ---
            new_svd_layer.u_proj.weight.data = truc_wu_cpu # u_proj 是 [out, rank]
            new_svd_layer.v_proj.weight.data = truc_wv_cpu # v_proj 是 [rank, in_svd] (in_svd = in_features - len(col_indices))
            if hasattr(new_svd_layer, 'col_proj'):
                new_svd_layer.col_proj.weight.data = col_w_cpu   # col_proj 是 [out, len(col_indices)]

            if original_layer.bias is not None:
                new_svd_layer.u_proj.bias.data = original_layer.bias.data.cpu() # 同样，如果需要，移动bias到CPU

            # 获取父模块并替换
            full_layer_name = f"{layer_prefix}.{i}.{name}"
            parent_module, module_name = get_parent_module_and_name(model, full_layer_name)
            setattr(parent_module, module_name, new_svd_layer)
            
            # --- Step 8: 内存清理 ---
            # 删除在当前循环迭代中创建且不再需要的张量
            del W, raw_scaling_diag_matrix, truc_wu, truc_wv, col_w, sigma_error_reduction, svd_error_reduction
            del truc_wu_cpu, truc_wv_cpu, col_w_cpu # 也清理CPU上的副本
            torch.cuda.empty_cache()

    # --- Step 9: 保存 Error Metrics 到文件 ---
    save_dir = 'error_metrics'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 使用 args.ratio 作为文件名中的目标压缩率
    ratio_target_str = f"{args.ratio:.2f}".replace('.', '_') # e.g., 0.70 -> 0_70
    model_name_suffix = model_name.split('/')[-1]
    save_name = f"{model_name_suffix}_{args.strategy}_ratio{ratio_target_str}_error_metrics.json"
    
    save_path = os.path.join(save_dir, save_name)
    print(f"Saving error metrics to: {save_path}")
    with open(save_path, "w") as f:
        json.dump(error_metrics, f, indent=4)

    print("模型所有指定层已成功替换为 CPSVD_Linear 层。Error metrics 已保存。")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='/cluster/home/xulin/programs/models/Llama-3.1-8B', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default='compressed_models/Llama-3.1-8B_cpsvd_0.8_dp', help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.8, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    
    # parser.add_argument('--hessian_mat_path', type=str, default='/cluster/home/xulin/programs/SkipCat/cache/whiten/Qwen3-8B_wikitext2_256_2048_3.pt', help='Local path to load the profiling matrices`')
    parser.add_argument('--hessian_mat_path', type=str, help='Local path to load the h matrices`')
    parser.add_argument('--seed',type=int, default=3, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--prefilling_len', type=int, default=256)

    parser.add_argument('--eval_ppl', action='store_true', default=True, help='whether to eval the model before saving')
    parser.add_argument('--eval_zero_shot', action='store_true', default=True, help='whether to eval the model before saving')
    parser.add_argument('--save_path', type=str, default='compressed_models', help='the path to save the compressed model checkpoints.`')
    
    parser.add_argument('--sensitivity_path', type=str)
    # parser.add_argument('--sensitivity_path', type=str, default='cache/Llama-3-8B_sensitivity_KL_around_0.8_32_wikitext2_seed3.pt')
    parser.add_argument('--strategy',type=str, default='dp', choices=['dp', 'greedy', 'uniform', 'strs', 'bolaco', 'sola'])
    parser.add_argument('--loss_type', type=str, default='kl')
    parser.add_argument('--min_ratio', type=float, default=0.1)
    parser.add_argument('--step', type=int, default=1, help='the step to run the compression')
    parser.add_argument('--cuda_devices', type=str, default='1', help='the cuda devices to run the model')
    
    args = parser.parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    # device_map = 'cpu' if args.run_low_resource else 'auto'
    # args.ratio = 1- args.ratio
    # step = -2: 将SVD-LLM的结果用标准形式保存 -1： 将ours的结果用标准形式保存 0：将ours的结果用
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='cpu')
        model = model.eval()
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        hessian_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
        torch.save(hessian_mat, args.save_path + "/" + args.model.split('/')[-1] + '_hessian_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')

    elif args.step == 0:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='cpu')
        # model, tokenizer = load_model_with_svd(args.model_path)
        model = model.eval()

        if args.hessian_mat_path is None:
            raise ValueError('No hessian mat path')
        else:
            hessian_mat = torch.load(args.hessian_mat_path)
        model = model.cpu()
        whitening(args.model, model, hessian_mat, args.ratio, args.DEV, args)
        
        if args.save_path is not None:
            # 1. Define a directory path for the model, not a single .pt file
            save_directory = (
                f"{args.save_path}/{args.model.split('/')[-1]}"
                f"_svd_llm_{args.ratio}_{args.strategy}"
            )

            # 2. Use the dedicated function to save the model and its custom config
            #    This function will call model.save_pretrained internally.
            save_model_with_svd_config(model, save_directory)

            # 3. Save the tokenizer to the same directory for a complete package
            tokenizer.save_pretrained(save_directory)
            config_path = os.path.join(save_directory, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 3. 修改 model_type
            model_type_old = config_data['model_type']
            config_data['model_type'] = 'cpsvd_' + model_type_old
            print(f"修改后 model_type: {config_data.get('model_type')}")

            # 4. 将修改后的内容写回文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"config.json 已成功更新！")
            print(f"✅ Model and tokenizer successfully saved to: {save_directory}")
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='auto')
        model = model.eval()
        # ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        hessian_mat = torch.load(args.hessian_mat_path)
        cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len, seed=args.seed)
        calib_sensitivity_kl(model, hessian_mat, cali_white_data, args)
        # calib_sensitivity_ppl(model, hessian_mat, cali_white_data, args)
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(model_id=args.model, device_map='cpu')
        # model, tokenizer = load_model_with_svd(args.model_path)
        model = model.eval()
        if args.hessian_mat_path is None:
            raise ValueError('No hessian mat path')
        else:
            hessian_mat = torch.load(args.hessian_mat_path)
        model = model.cpu()
        whitening_cp(args.model, model, hessian_mat, args.ratio, args.DEV, args)

        if args.save_path is not None:
            # 1. Define a directory path for the model, not a single .pt file
            save_directory = (
                f"{args.save_path}/{args.model.split('/')[-1]}"
                f"_cpsvd_{args.ratio}"
                f'_{args.strategy}'
            )

            # 2. Use the dedicated function to save the model and its custom config
            #    This function will call model.save_pretrained internally.
            save_model_with_svd_config(model, save_directory)

            # 3. Save the tokenizer to the same directory for a complete package
            tokenizer.save_pretrained(save_directory)
            config_path = os.path.join(save_directory, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 3. 修改 model_type
            model_type_old = config_data['model_type']
            config_data['model_type'] = 'cpsvd_' + model_type_old
            print(f"修改后 model_type: {config_data.get('model_type')}")

            # 4. 将修改后的内容写回文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            print(f"config.json 已成功更新！")
            print(f"✅ Model and tokenizer successfully saved to: {save_directory}")
    
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        # if args.step == 5:
        #     model, tokenizer = get_model_from_local(args.model_path)
        #     model = model.to(torch.float16)
        #     model = model.to('cuda')
        #     # model, tokenizer = get_model_from_huggingface(args.model)
        # else:
        #     # model, tokenizer = get_model_from_huggingface(args.model_path, device_map='auto')
        #     model, tokenizer = get_model_from_huggingface(args.model_path, device_map='auto')
        model, tokenizer = get_model_from_huggingface(args.model_path, device_map='auto')
        model.eval()
        # model = model.float()
        # model = model.to(args.DEV)
        # model = dispatch_model(model, device_map="auto")
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=4096, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            # eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
            prefilling_decoding_eval(model, tokenizer, original_len=args.prefilling_len, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 6:
            hflm = HFLM(pretrained=model, tokenizer=tokenizer)
            task_list = ['mmlu']
            res = lm_eval.simple_evaluate(hflm, tasks=task_list, num_fewshot=5)
            print(res['results'])
            # hflm = HFLM(pretrained=args.model_path, tokenizer=args.model_path, dtype=torch.float16)
            task_list = ["hellaswag", "arc_challenge", "winogrande", "arc_easy", "openbookqa", "piqa"]
            res = lm_eval.simple_evaluate(hflm, tasks=task_list, num_fewshot=0)
            print(res['results'])



