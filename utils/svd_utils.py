import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn

from model_utils import find_layers

from evaluater import evaluate_perplexity

def hessian_weight_svd(W, raw_scaling_diag_matrix, dev, scale_float=False, svd_double=False):
    raw_scaling_diag_matrix = raw_scaling_diag_matrix.to(dev).double()
    try:
        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
    except Exception as e:
        print("Warning: eigen scaling_diag_matrix is not positive!")
        # eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
        # raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
        damp = 0.05 * torch.mean(torch.diag(raw_scaling_diag_matrix))
        n = raw_scaling_diag_matrix.size(0)
        idx = torch.arange(n, device=raw_scaling_diag_matrix.device)
        raw_scaling_diag_matrix[idx, idx] += damp
        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
        eigenvalues = raw_scaling_diag_matrix = None
        del eigenvalues
    try:
        if scale_float:
            scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not full rank!")
        scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(dev)
        scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
    if not svd_double:
        scaling_diag_matrix = scaling_diag_matrix.float()
        scaling_matrix_inv = scaling_matrix_inv.float()
    else:
        W = W.double()
    W_scale = torch.matmul(W, scaling_diag_matrix)
    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

    if svd_double:
        return U.float(), S.float(), VT.float(), scaling_matrix_inv.float()
    else:
        return U, S, VT, scaling_matrix_inv
    
def hessian_weight_svd_v2(W, H, double=False):
    if double:
        H = H.to(W.device).double()
        W = W.double()
    else:
        H = H.to(W.device).float()
    Uh, Sh, VhT = torch.linalg.svd(H, full_matrices=False)
    Sx = torch.diag(Sh.sqrt())
    Wh = W@Uh@Sx
    Uwh, Swh, VwhT = torch.linalg.svd(Wh, full_matrices=False)

    Sx_inv = torch.diag(Sh.sqrt().reciprocal())
    Uh_inv = Uh.T
    sqrt_Sigma = torch.diag(Swh.sqrt()) 
    W_u = Uwh @ sqrt_Sigma 
    W_v = sqrt_Sigma @ VwhT @ Sx_inv @ Uh_inv
    return W_u.float(), W_v.float(), Swh.float()

# @torch.no_grad()
# def calib_sensitivity_ppl(model, hessian_mat, calib_loader, args):
#     model_id = model.config._name_or_path.split('/')[-1]
#     cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_ppl_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
#     if os.path.exists(cache_file):
#         sensitivity_dict = torch.load(cache_file, map_location="cpu")
#         return sensitivity_dict

#     model.eval()

#     sensitivity_dict = {}
#     param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
#     ppl_base = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
#     layers = model.model.layers
#     pbar = tqdm(total=len(find_layers(layers[0])) * len(param_ratio_candidates) * len(layers))
#     for i in range(len(layers)):
#         layer = layers[i]
#         # sensitivity_dict[i] = {}
#         subset = find_layers(layer)
#         for name in subset:
#             full_name = f'model.layers.{i}.{name}'
#             sensitivity_dict[full_name] = {}
#             sensitivity_dict[full_name][1] = ppl_base

#             W = subset[name].weight.data.float()
#             dtype = subset[name].weight.data.dtype
#             raw_scaling_diag_matrix = hessian_mat[i][name]
#             scale_float = True if 'Qwen3' in args.model else False
#             U, S, VT, scaling_matrix_inv = hessian_weight_svd(W, raw_scaling_diag_matrix, W.device, scale_float)

#             for param_ratio in param_ratio_candidates:
#                 num_s_after_trunc = int(W.shape[0] * W.shape[1] * param_ratio / (W.shape[0] + W.shape[1]))
#                 truc_s = S[:num_s_after_trunc]
#                 truc_u = U[:, :num_s_after_trunc]
#                 truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
#                 truc_sigma = torch.diag(truc_s)
#                 #### Replace Attn, MLP ####
#                 sqrtSigma = torch.sqrt(truc_sigma)
#                 svd_u = torch.matmul(truc_u, sqrtSigma)
#                 svd_v = torch.matmul(sqrtSigma, truc_v)
#                 subset[name].weight.data = (svd_u@svd_v).to(dtype)
#                 ppl = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
#                 full_name = f'model.layers.{i}.{name}'
#                 sensitivity_dict[full_name][param_ratio] = ppl
#                 pbar.update(1)
#                 # print(f"{info['full_name']} {param_ratio} {ppl}")
#             subset[name].weight.data = W.to(dtype)
#             W = scaling_diag_matrix = scaling_matrix_inv = W_scale = U = S = VT = svd_u = svd_v = truc_s = truc_u = truc_v = truc_sigma = sqrtSigma = None
#             torch.cuda.empty_cache() 
#         # setattr(info["father"], info["name"], raw_linear)
        
#     torch.save(sensitivity_dict, cache_file)
#     return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_div(model, hessian_mat, calib_loader, args):
    """
    Calculates the sensitivity of each module by measuring the divergence of the 
    final layer's hidden state after pruning. This version includes aggressive VRAM
    management and calculates L1, L2, and Cosine divergence using float32 for precision.
    """
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_div_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    
    if os.path.exists(cache_file):
        print(f"Loading cached sensitivity results from: {cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()

    # --- Step 1: Pre-compute original hidden states ---
    print("Pre-computing original hidden states for all calibration batches (on GPU)...")
    original_hidden_states_gpu = []
    for batch in tqdm(calib_loader, desc="Pre-computation"):
        input_ids = batch["input_ids"].to(args.DEV)
        outputs_original = model(input_ids, output_hidden_states=True)
        h_original_batch = outputs_original.hidden_states[-1]
        original_hidden_states_gpu.append(h_original_batch)
    print("Pre-computation complete.")

    del outputs_original
    torch.cuda.empty_cache()

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    layers = model.model.layers
    
    total_iterations = len(find_layers(layers[0])) * len(param_ratio_candidates) * len(layers)
    pbar = tqdm(total=total_iterations, desc="Calculating Divergence")
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            full_name = f'model.layers.{i}.{name}'
            sensitivity_dict[full_name] = {}
            sensitivity_dict[full_name][1.0] = {'l1': 0.0, 'l2': 0.0, 'cos': 0.0}

            W_orig = subset[name].weight.data.clone()
            dtype = W_orig.dtype
            
            raw_scaling_diag_matrix = hessian_mat[i][name]
            
            scale_float = True if 'Qwen3' in args.model else False
            U, S, VT, scaling_matrix_inv = hessian_weight_svd(W_orig.float(), raw_scaling_diag_matrix, W_orig.device, scale_float)

            for param_ratio in param_ratio_candidates:
                num_s_after_trunc = int(W_orig.shape[0] * W_orig.shape[1] * param_ratio / (W_orig.shape[0] + W_orig.shape[1]))
                if num_s_after_trunc < 1:
                    num_s_after_trunc = 1
                
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                
                W_pruned = (svd_u @ svd_v).to(dtype)
                subset[name].weight.data = W_pruned

                del truc_s, truc_u, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v, W_pruned
                
                total_l1_div = 0.0
                total_l2_div = 0.0
                total_cos_div = 0.0
                
                for batch_idx, batch in enumerate(calib_loader):
                    input_ids = batch["input_ids"].to(args.DEV)
                    h_original_batch = original_hidden_states_gpu[batch_idx]

                    outputs_pruned = model(input_ids, output_hidden_states=True)
                    h_pruned = outputs_pruned.hidden_states[-1]

                    # --- MODIFICATION START: Cast to float32 for high-precision calculation ---
                    h_original_float = h_original_batch.float()
                    h_pruned_float = h_pruned.float()
                    
                    diff = h_original_float - h_pruned_float
                    
                    # Metric 1: L2 Divergence (Mean Squared Error)
                    l2_div_batch = torch.mean(torch.sum(diff.pow(2), dim=-1))
                    
                    # Metric 2: L1 Divergence (Mean Absolute Error)
                    l1_div_batch = torch.mean(torch.sum(diff.abs(), dim=-1))

                    # Metric 3: Cosine Distance
                    cos_sim_batch = F.cosine_similarity(h_original_float, h_pruned_float, dim=-1)
                    cos_dist_batch = torch.mean(1.0 - cos_sim_batch)
                    # --- MODIFICATION END ---

                    total_l2_div += l2_div_batch.item()
                    total_l1_div += l1_div_batch.item()
                    total_cos_div += cos_dist_batch.item()

                avg_l1_div = total_l1_div / len(calib_loader)
                avg_l2_div = total_l2_div / len(calib_loader)
                avg_cos_div = total_cos_div / len(calib_loader)
                
                sensitivity_dict[full_name][param_ratio] = {
                    'l1': avg_l1_div,
                    'l2': avg_l2_div,
                    'cos': avg_cos_div
                }
                
                pbar.update(1)

            subset[name].weight.data = W_orig
            
            del W_orig, U, S, VT, scaling_matrix_inv
            torch.cuda.empty_cache()
                
    pbar.close()
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving sensitivity results to cache file: {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    del original_hidden_states_gpu
    torch.cuda.empty_cache()
    
    return sensitivity_dict


# @torch.no_grad()
# def calib_sensitivity_kl(model, hessian_mat, calib_loader, args):
#     """
#     Calculates the sensitivity of each module by measuring the KL Divergence of the 
#     final output probability distribution after pruning.
    
#     This version ensures the KL divergence calculation is performed in float32 for
#     high precision, even if the model runs in float16/bfloat16.
#     """
#     model_id = model.config._name_or_path.split('/')[-1]
#     # Changed cache file name to reflect the new metric (KL divergence)
#     cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_KL_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    
#     if os.path.exists(cache_file):
#         print(f"Loading cached sensitivity results from: {cache_file}")
#         sensitivity_dict = torch.load(cache_file, map_location="cpu")
#         return sensitivity_dict

#     model.eval()

#     # --- Step 1: Pre-compute original logits and store on CPU ---
#     print("Pre-computing original logits for all calibration batches (and moving to CPU)...")
#     original_logits_cpu = []
#     for batch in tqdm(calib_loader, desc="Pre-computation"):
#         input_ids = batch["input_ids"].to(args.DEV)
#         outputs_original = model(input_ids)
#         original_logits_cpu.append(outputs_original.logits.cpu())
#     print("Pre-computation complete.")

#     del outputs_original
#     torch.cuda.empty_cache()

#     sensitivity_dict = {}
#     param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     layers = model.model.layers
    
#     total_iterations = len(find_layers(layers[0])) * len(param_ratio_candidates) * len(layers)
#     pbar = tqdm(total=total_iterations, desc="Calculating KL Divergence")
    
#     kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)

#     # --- Step 2: Main loop to prune each module and calculate divergence ---
#     for i in range(len(layers)):
#         layer = layers[i]
#         subset = find_layers(layer)
#         for name in subset:
#             full_name = f'model.layers.{i}.{name}'
#             sensitivity_dict[full_name] = {}
#             sensitivity_dict[full_name][1.0] = 0.0

#             W_orig = subset[name].weight.data.clone()
#             dtype = W_orig.dtype
            
#             raw_scaling_diag_matrix = hessian_mat[i][name]
            
#             scale_float = True if 'Qwen3' in args.model else False
#             U, S, VT, scaling_matrix_inv = hessian_weight_svd(W_orig.float(), raw_scaling_diag_matrix, W_orig.device, scale_float)

#             for param_ratio in param_ratio_candidates:
#                 num_s_after_trunc = int(W_orig.shape[0] * W_orig.shape[1] * param_ratio / (W_orig.shape[0] + W_orig.shape[1]))
#                 if num_s_after_trunc < 1:
#                     num_s_after_trunc = 1
                
#                 truc_s = S[:num_s_after_trunc]
#                 truc_u = U[:, :num_s_after_trunc]
#                 truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
#                 truc_sigma = torch.diag(truc_s)
#                 sqrtSigma = torch.sqrt(truc_sigma)
#                 svd_u = torch.matmul(truc_u, sqrtSigma)
#                 svd_v = torch.matmul(sqrtSigma, truc_v)
                
#                 W_pruned = (svd_u @ svd_v).to(dtype)
#                 subset[name].weight.data = W_pruned

#                 del truc_s, truc_u, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v, W_pruned
                
#                 # --- Step 3: Calculate divergence batch-by-batch ---
#                 total_kl_divergence = 0.0
#                 for batch_idx, batch in enumerate(calib_loader):
#                     input_ids = batch["input_ids"].to(args.DEV)
                    
#                     original_logits_batch = original_logits_cpu[batch_idx].to(args.DEV)

#                     outputs_pruned = model(input_ids)
#                     pruned_logits_batch = outputs_pruned.logits

#                     # --- MODIFICATION START: Cast logits to float32 for high-precision calculation ---
#                     pruned_logits_float = pruned_logits_batch.float()
#                     original_logits_float = original_logits_batch.float()
                    
#                     # The input to KLDivLoss should be log-probabilities
#                     log_probs_pruned = F.log_softmax(pruned_logits_float, dim=-1)
#                     # The target can also be log-probabilities (more stable)
#                     log_probs_original = F.log_softmax(original_logits_float, dim=-1)
                    
#                     kl_div_batch = kl_loss_fn(log_probs_pruned, log_probs_original)
#                     # --- MODIFICATION END ---

#                     total_kl_divergence += kl_div_batch.item()
                    
#                     del original_logits_batch, outputs_pruned, pruned_logits_batch, kl_div_batch

#                 avg_kl_divergence = total_kl_divergence / len(calib_loader)
#                 sensitivity_dict[full_name][param_ratio] = avg_kl_divergence
                
#                 pbar.update(1)

#             subset[name].weight.data = W_orig
            
#             del W_orig, U, S, VT, scaling_matrix_inv
#             torch.cuda.empty_cache()
                
#     pbar.close()
    
#     os.makedirs(os.path.dirname(cache_file), exist_ok=True)
#     print(f"Saving sensitivity results to cache file: {cache_file}")
#     torch.save(sensitivity_dict, cache_file)
    
#     del original_logits_cpu
#     torch.cuda.empty_cache()
    
#     return sensitivity_dict


import gc # Garbage Collector

# =======================================================================================
# Assume find_layers and hessian_weight_svd functions are defined elsewhere
# =======================================================================================

@torch.no_grad()
def calib_sensitivity_kl(model, hessian_mat, calib_loader, args):
    """
    Calculates the sensitivity of each module by measuring the KL Divergence.
    
    VRAM Optimized Logic (Device Alignment Aware):
    1. [Optimization] 'backup' of the current layer is moved to CPU to save VRAM.
    2. [Optimization] SVD is computed ONCE per layer. U, S, VT are moved to CPU.
    3. [Optimization] Slicing happens on CPU. Only small slices move to GPU for reconstruction.
    """
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_KL_around_{args.ratio}_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    
    if os.path.exists(cache_file):
        print(f"Loading cached sensitivity results from: {cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()
    
    base_prune_ratio = args.ratio
    if not (0 < base_prune_ratio < 1):
        raise ValueError(f"args.ratio must be between (0, 1), but got: {base_prune_ratio}")

    # --- Step 1: Pre-compute original logits (ratio=1.0) and store on CPU ---
    print("Pre-computing original logits (ratio=1.0) for all calibration batches (CPU)...")
    original_logits_cpu = []
    
    logit_computation_device = next(model.parameters()).device # Get device from model
    for batch in tqdm(calib_loader, desc="Pre-computation (Original Logits)"):
        input_ids = batch["input_ids"].to(logit_computation_device) 
        outputs_original = model(input_ids)
        original_logits_cpu.append(outputs_original.logits.cpu()) 
    print("Pre-computation complete.")
    del outputs_original; torch.cuda.empty_cache()

    # --- Step 1.5: Pre-compute weights: 1.0 on CPU, base_ratio set directly on target device ---
    print(f"Pre-computing weights and setting base pruned state (all modules @ {base_prune_ratio} ratio)...")
    W_orig_1_0_map_cpu = {}     # Stores original weights (1.0) on CPU
    
    all_layers_subset = {}
    layers = model.model.layers
    for i in range(len(layers)):
        all_layers_subset.update({f'model.layers.{i}.{name}': layer for name, layer in find_layers(layers[i]).items()})

    for full_name, layer_module in tqdm(all_layers_subset.items(), desc="Pre-computing Base Weights"):
        if not hasattr(layer_module, 'weight'): continue

        target_device = layer_module.weight.device
        i, name = int(full_name.split('.')[2]), full_name.split('.')[-2] + '.' + full_name.split('.')[-1]
        
        W_orig_target_device = layer_module.weight.data 
        dtype = W_orig_target_device.dtype
        W_orig_1_0_map_cpu[full_name] = W_orig_target_device.clone().cpu() 
        
        if i not in hessian_mat or name not in hessian_mat[i]:
             print(f"Warning: Hessian info not found for {full_name}. Skipping base pruning pre-computation.")
             continue

        raw_scaling_diag_matrix = hessian_mat[i][name]
        scale_float = 'Qwen3' in args.model
        
        # Perform SVD on the target device
        U, S, VT, scaling_matrix_inv = hessian_weight_svd(W_orig_target_device.float(), raw_scaling_diag_matrix, target_device, scale_float)
        
        num_s_base = int(W_orig_target_device.shape[0] * W_orig_target_device.shape[1] * base_prune_ratio / (W_orig_target_device.shape[0] + W_orig_target_device.shape[1]))
        if num_s_base < 1: num_s_base = 1
        
        truc_s = S[:num_s_base]; truc_u = U[:, :num_s_base]; truc_v = torch.matmul(VT[:num_s_base, :], scaling_matrix_inv)
        truc_sigma = torch.diag(truc_s); sqrtSigma = torch.sqrt(truc_sigma)
        svd_u = torch.matmul(truc_u, sqrtSigma); svd_v = torch.matmul(sqrtSigma, truc_v)
        
        W_pruned_base_target_device = (svd_u @ svd_v).to(device=target_device, dtype=dtype)
        
        layer_module.weight.data = W_pruned_base_target_device
        
        del U, S, VT, scaling_matrix_inv, truc_s, truc_u, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v
    
    print("Base pruned state pre-computed and set. Original weights backup is on CPU.")
    torch.cuda.empty_cache(); gc.collect()

    kl_loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=True)

    # --- Step 1.6: Calculate KL divergence for the global base pruned state ONCE ---
    print(f"Calculating KL divergence for the base pruned state (ratio={base_prune_ratio}) once...")
    kl_divergence_base_ratio = 0.0
    for batch_idx, batch in enumerate(tqdm(calib_loader, desc=f"KL Calc (Ratio={base_prune_ratio})")):
        input_ids = batch["input_ids"].to(args.DEV)
        original_logits_batch = original_logits_cpu[batch_idx].to(args.DEV)
        
        outputs_pruned = model(input_ids)
        pruned_logits_batch = outputs_pruned.logits

        pruned_logits_float = pruned_logits_batch.float()
        original_logits_float = original_logits_batch.float()
        log_probs_pruned = F.log_softmax(pruned_logits_float, dim=-1)
        log_probs_original = F.log_softmax(original_logits_float, dim=-1)
        
        kl_div_batch = kl_loss_fn(log_probs_pruned, log_probs_original)
        kl_divergence_base_ratio += kl_div_batch.item()
        
        del original_logits_batch, outputs_pruned, pruned_logits_batch, kl_div_batch

    kl_divergence_base_ratio /= len(calib_loader)
    print(f"KL divergence for base ratio {base_prune_ratio}: {kl_divergence_base_ratio:.6f}")
    torch.cuda.empty_cache()

    # --- Step 2: Main loop for sensitivity analysis ---
    param_ratio_candidates = sorted(list(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, base_prune_ratio])))
    
    sensitivity_dict = {}
    
    pbar = tqdm(total=len(all_layers_subset) * len(param_ratio_candidates), desc="Calculating Sensitivity KL")

    for full_name, layer_module in all_layers_subset.items():
        if not hasattr(layer_module, 'weight') or full_name not in W_orig_1_0_map_cpu: 
             pbar.update(len(param_ratio_candidates)); continue

        target_device = layer_module.weight.device
        sensitivity_dict[full_name] = {}
        
        W_orig_cpu = W_orig_1_0_map_cpu[full_name]
        dtype = W_orig_cpu.dtype
        i, name = int(full_name.split('.')[2]), full_name.split('.')[-2] + '.' + full_name.split('.')[-1]
        
        if i not in hessian_mat or name not in hessian_mat[i]:
            print(f"Warning: Hessian info not found for {full_name} during main loop. Skipping.")
            for pr in param_ratio_candidates: sensitivity_dict[full_name][pr] = float('nan')
            pbar.update(len(param_ratio_candidates)); continue

        raw_scaling_diag_matrix = hessian_mat[i][name]
        scale_float = 'Qwen3' in args.model

        # [VRAM 优化 1] 将当前层的 Base 状态备份到 CPU，而不是 GPU
        W_pruned_base_cpu_backup = layer_module.weight.data.clone().cpu()

        # [VRAM 优化 2] 预先计算 SVD 并将组件移动到 CPU
        # 临时将原始权重移回 GPU 计算 SVD (速度最快)
        W_orig_gpu_temp = W_orig_cpu.to(target_device)
        U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu = hessian_weight_svd(W_orig_gpu_temp.float(), raw_scaling_diag_matrix, target_device, scale_float)
        
        # 立即将分解结果移到 CPU 保存，并释放 GPU 显存
        U_cpu = U_gpu.cpu()
        S_cpu = S_gpu.cpu()
        VT_cpu = VT_gpu.cpu()
        scaling_matrix_inv_cpu = scaling_matrix_inv_gpu.cpu()
        
        del W_orig_gpu_temp, U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu
        torch.cuda.empty_cache() # 清理 GPU 碎片

        rows = W_orig_cpu.shape[0]
        cols = W_orig_cpu.shape[1]

        for param_ratio in param_ratio_candidates:
            
            if param_ratio == base_prune_ratio:
                sensitivity_dict[full_name][param_ratio] = kl_divergence_base_ratio
                pbar.update(1); continue 

            W_pruned_new_target_device = None
            if param_ratio == 1.0:
                W_pruned_new_target_device = W_orig_cpu.to(target_device)
                
            else: 
                # [VRAM 优化 3] CPU Slicing: 只计算切片并移动小块到 GPU
                num_s = int(rows * cols * param_ratio / (rows + cols))
                if num_s < 1: num_s = 1
                
                # Slicing on CPU
                truc_s_cpu = S_cpu[:num_s]
                truc_u_cpu = U_cpu[:, :num_s]
                truc_vt_slice_cpu = VT_cpu[:num_s, :]
                
                # Move slices to Target Device
                truc_s = truc_s_cpu.to(target_device)
                truc_u = truc_u_cpu.to(target_device)
                truc_vt_slice = truc_vt_slice_cpu.to(target_device)
                scaling_inv = scaling_matrix_inv_cpu.to(target_device)

                # Reconstruction on Target Device
                # Logic: truc_v = VT_slice @ scaling_inv
                truc_v = torch.matmul(truc_vt_slice, scaling_inv)
                truc_sigma = torch.diag(truc_s)
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                
                W_pruned_new_target_device = (svd_u @ svd_v).to(device=target_device, dtype=dtype)
                
                del truc_s, truc_u, truc_vt_slice, scaling_inv, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v
            
            # 赋值 (Source and Target are on same device)
            layer_module.weight.data = W_pruned_new_target_device

            # --- Calculate KL Divergence ---
            total_kl_divergence = 0.0
            for batch_idx, batch in enumerate(calib_loader):
                # 确保 input 和 logits 都在做前向传播的那个主设备上 (args.DEV)
                input_ids = batch["input_ids"].to(args.DEV)
                original_logits_batch = original_logits_cpu[batch_idx].to(args.DEV) 
                
                outputs_pruned = model(input_ids)
                pruned_logits_batch = outputs_pruned.logits

                pruned_logits_float = pruned_logits_batch.float()
                original_logits_float = original_logits_batch.float()
                log_probs_pruned = F.log_softmax(pruned_logits_float, dim=-1)
                log_probs_original = F.log_softmax(original_logits_float, dim=-1)
                
                kl_div_batch = kl_loss_fn(log_probs_pruned, log_probs_original)
                total_kl_divergence += kl_div_batch.item()
                
                del original_logits_batch, outputs_pruned, pruned_logits_batch, kl_div_batch

            avg_kl_divergence = total_kl_divergence / len(calib_loader)
            sensitivity_dict[full_name][param_ratio] = avg_kl_divergence
            
            # 及时清理生成的权重
            del W_pruned_new_target_device
            pbar.update(1)
            if param_ratio != 1.0: torch.cuda.empty_cache()

        # 循环结束，清理 CPU 上的 SVD 组件
        del U_cpu, S_cpu, VT_cpu, scaling_matrix_inv_cpu

        # [VRAM 优化 1 后续] 从 CPU 恢复 Base 状态
        layer_module.weight.data = W_pruned_base_cpu_backup.to(target_device)
        del W_pruned_base_cpu_backup 
        torch.cuda.empty_cache() 
            
    pbar.close()
    
    # --- Step 5: Restore the model to its original (ratio=1.0) weights ---
    print("Sensitivity analysis complete. Restoring original model weights...")
    for full_name, module in all_layers_subset.items():
        if full_name in W_orig_1_0_map_cpu:
            target_device = module.weight.device
            module.weight.data = W_orig_1_0_map_cpu[full_name].to(target_device)
    print("Model restored to original state.")
    
    # --- Step 6: Save results and final cleanup ---
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving sensitivity results to cache file: {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    del original_logits_cpu, W_orig_1_0_map_cpu
    torch.cuda.empty_cache()
    gc.collect() 
    
    return sensitivity_dict



@torch.no_grad()
def calib_sensitivity_ppl(model, hessian_mat, calib_loader, args):
    """
    Calculates sensitivity by measuring Perplexity (PPL).
    
    VRAM Optimized Logic (Enhanced):
    1. Base pruned state calculation remains similar.
    2. [Optimization] 'backup' of the current layer is moved to CPU to save VRAM.
    3. [Optimization] SVD is computed ONCE per layer. U, S, VT are moved to CPU.
    4. [Optimization] Slicing happens on CPU. Only small slices move to GPU for reconstruction.
    """
    
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_ppl_base_ratio_{args.ratio}_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    
    if os.path.exists(cache_file):
        print(f"Loading cached sensitivity results from: {cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()
    
    base_prune_ratio = args.ratio
    if not (0 < base_prune_ratio < 1):
        raise ValueError(f"args.ratio must be between (0, 1), but got: {base_prune_ratio}")

    print("Concatenating all calibration data for PPL evaluation...")
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print("Data concatenation complete.")

    # --- 步骤 1.5: 预计算权重并设置基础压缩状态 ---
    print(f"Pre-computing weights and setting base pruned state (all modules @ {base_prune_ratio} ratio)...")
    W_orig_1_0_map_cpu = {}       
    
    all_layers_subset = {}
    layers = model.model.layers
    for i in range(len(layers)):
        all_layers_subset.update({f'model.layers.{i}.{name}': layer for name, layer in find_layers(layers[i]).items()})

    for full_name, layer_module in tqdm(all_layers_subset.items(), desc="Pre-computing Base Weights"):
        if not hasattr(layer_module, 'weight'): continue

        target_device = layer_module.weight.device
        i, name = int(full_name.split('.')[2]), full_name.split('.')[-2] + '.' + full_name.split('.')[-1]
        
        # 保持原始权重在 CPU
        W_orig_target_device = layer_module.weight.data
        dtype = W_orig_target_device.dtype
        W_orig_1_0_map_cpu[full_name] = W_orig_target_device.clone().cpu()
        
        if i not in hessian_mat or name not in hessian_mat[i]:
            print(f"Warning: Hessian info not found for {full_name}. Skipping base pruning pre-computation.")
            continue

        raw_scaling_diag_matrix = hessian_mat[i][name]
        scale_float = 'Qwen3' in args.model
        
        # 计算 SVD
        U, S, VT, scaling_matrix_inv = hessian_weight_svd(W_orig_target_device.float(), raw_scaling_diag_matrix, target_device, scale_float)
        
        # 计算截断数量
        num_s_base = int(W_orig_target_device.shape[0] * W_orig_target_device.shape[1] * base_prune_ratio / (W_orig_target_device.shape[0] + W_orig_target_device.shape[1]))
        if num_s_base < 1: num_s_base = 1
        
        # 截断并重构 (均在 GPU 进行，计算完立即释放)
        truc_s = S[:num_s_base]; truc_u = U[:, :num_s_base]; truc_v = torch.matmul(VT[:num_s_base, :], scaling_matrix_inv)
        truc_sigma = torch.diag(truc_s); sqrtSigma = torch.sqrt(truc_sigma)
        svd_u = torch.matmul(truc_u, sqrtSigma); svd_v = torch.matmul(sqrtSigma, truc_v)
        
        W_pruned_base_target_device = (svd_u @ svd_v).to(device=target_device, dtype=dtype)
        
        layer_module.weight.data = W_pruned_base_target_device
        
        del U, S, VT, scaling_matrix_inv, truc_s, truc_u, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v, W_pruned_base_target_device
    
    print("Base pruned state pre-computed and set.")
    torch.cuda.empty_cache(); gc.collect()

    # --- 步骤 1.6: 计算全局基础压缩状态的 PPL (仅一次) ---
    print(f"Calculating PPL for the base pruned state (ratio={base_prune_ratio}) once...")
    ppl_base = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
    print(f"PPL for base ratio {base_prune_ratio}: {ppl_base:.6f}")
    torch.cuda.empty_cache()

    # --- 步骤 2: 敏感度分析主循环 ---
    param_ratio_candidates = sorted(list(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, base_prune_ratio])))
    
    sensitivity_dict = {}
    
    pbar = tqdm(total=len(all_layers_subset) * len(param_ratio_candidates), desc="Calculating Sensitivity PPL")

    for full_name, layer_module in all_layers_subset.items():
        if not hasattr(layer_module, 'weight') or full_name not in W_orig_1_0_map_cpu: 
            pbar.update(len(param_ratio_candidates)); continue

        target_device = layer_module.weight.device
        sensitivity_dict[full_name] = {}
        
        W_orig_cpu = W_orig_1_0_map_cpu[full_name]
        dtype = W_orig_cpu.dtype
        i, name = int(full_name.split('.')[2]), full_name.split('.')[-2] + '.' + full_name.split('.')[-1]
        
        if i not in hessian_mat or name not in hessian_mat[i]:
            for pr in param_ratio_candidates: sensitivity_dict[full_name][pr] = float('nan')
            pbar.update(len(param_ratio_candidates)); continue

        # [VRAM 优化 1] 将当前层的 Base 状态备份到 CPU，而不是 GPU
        # 这样在运行 evaluate_perplexity 时，GPU 上少了一份权重副本
        W_pruned_base_cpu_backup = layer_module.weight.data.clone().cpu()

        # [VRAM 优化 2] 预先计算 SVD 并将组件移动到 CPU
        # 避免在内部循环中重复计算 SVD，避免 SVD 中间变量占用 GPU
        raw_scaling_diag_matrix = hessian_mat[i][name]
        scale_float = 'Qwen3' in args.model
        
        # 将原始权重临时移回 GPU 进行 SVD 分解 (速度快)
        W_orig_gpu_temp = W_orig_cpu.to(target_device)
        U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu = hessian_weight_svd(W_orig_gpu_temp.float(), raw_scaling_diag_matrix, target_device, scale_float)
        
        # 立即将分解结果移到 CPU 保存，并释放 GPU 显存
        U_cpu = U_gpu.cpu()
        S_cpu = S_gpu.cpu()
        # 注意：这里我们提前把 VT 和 scaling_inv 乘好或者分别存，原逻辑是 matmul 后再存
        # 为了方便切片，我们分别存，或者如果显存允许，也可以先处理 VT
        VT_cpu = VT_gpu.cpu()
        scaling_matrix_inv_cpu = scaling_matrix_inv_gpu.cpu()
        
        del W_orig_gpu_temp, U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu
        torch.cuda.empty_cache() # 确保 SVD 产生的碎片被清理

        rows = W_orig_cpu.shape[0]
        cols = W_orig_cpu.shape[1]

        for param_ratio in param_ratio_candidates:
            if param_ratio == base_prune_ratio:
                sensitivity_dict[full_name][param_ratio] = ppl_base
                pbar.update(1); continue
            
            # 构造新权重
            W_pruned_new_target_device = None
            if param_ratio == 1.0:
                W_pruned_new_target_device = W_orig_cpu.to(target_device)
            else:
                # [VRAM 优化 3] 在 CPU 上计算切片索引，只移动需要的切片到 GPU
                num_s = int(rows * cols * param_ratio / (rows + cols))
                if num_s < 1: num_s = 1
                
                # CPU Slicing
                truc_s_cpu = S_cpu[:num_s]
                truc_u_cpu = U_cpu[:, :num_s]
                # VT 需要特殊处理，看原逻辑: truc_v = torch.matmul(VT[:num_s_base, :], scaling_matrix_inv)
                # 我们可以把 VT 切片后移到 GPU，scaling_inv 也移到 GPU，在 GPU 上乘 (或者都在 CPU 上乘)
                # 为了速度平衡，建议将切片后的分量移到 GPU 做乘法
                
                # Move slices to GPU
                truc_s = truc_s_cpu.to(target_device)
                truc_u = truc_u_cpu.to(target_device)
                truc_vt_slice = VT_cpu[:num_s, :].to(target_device)
                scaling_inv = scaling_matrix_inv_cpu.to(target_device)
                
                # Reconstruction on GPU
                truc_v = torch.matmul(truc_vt_slice, scaling_inv)
                truc_sigma = torch.diag(truc_s)
                sqrtSigma = torch.sqrt(truc_sigma)
                svd_u = torch.matmul(truc_u, sqrtSigma)
                svd_v = torch.matmul(sqrtSigma, truc_v)
                
                W_pruned_new_target_device = (svd_u @ svd_v).to(device=target_device, dtype=dtype)
                
                # 清理重建过程的临时变量
                del truc_s, truc_u, truc_vt_slice, scaling_inv, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v
            
            # 赋值并评测
            layer_module.weight.data = W_pruned_new_target_device
            
            ppl = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
            sensitivity_dict[full_name][param_ratio] = ppl
            
            pbar.update(1)
            
            # 这里的 del 很重要，释放当前比例的权重
            del W_pruned_new_target_device
            if param_ratio != 1.0: torch.cuda.empty_cache()

        # 循环结束，清理 CPU 上的 SVD 组件
        del U_cpu, S_cpu, VT_cpu, scaling_matrix_inv_cpu

        # [VRAM 优化 1 后续] 从 CPU 恢复 Base 状态
        layer_module.weight.data = W_pruned_base_cpu_backup.to(target_device)
        del W_pruned_base_cpu_backup
        torch.cuda.empty_cache()
        
    pbar.close()

    # --- 步骤 5: 恢复模型到原始 (1.0) 状态 ---
    print("Sensitivity analysis complete. Restoring original model weights...")
    for full_name, module in all_layers_subset.items():
        if full_name in W_orig_1_0_map_cpu:
            target_device = module.weight.device
            module.weight.data = W_orig_1_0_map_cpu[full_name].to(target_device)
    print("Model restored to original state.")
    
    # --- 步骤 6: 保存和清理 ---
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving sensitivity results to cache file: {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    del W_orig_1_0_map_cpu, input_ids, all_layers_subset
    torch.cuda.empty_cache()
    gc.collect()
    
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_ppl_asvd(model, hessian_mat, calib_loader, args):
    """
    Calculates sensitivity by measuring Perplexity (PPL).
    
    Optimized Logic:
    1. No global backup. Backups are done layer-by-layer (JIT) to save RAM.
    2. Workflow: Backup Layer X -> Prune Layer X -> Eval -> Restore Layer X.
    """
    
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_ppl_asvd_{args.whitening_nsamples}_{args.dataset}_seed{args.seed}.pt"
    
    if os.path.exists(cache_file):
        print(f"Loading cached sensitivity results from: {cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict

    model.eval()
    
    print("Concatenating all calibration data for PPL evaluation...")
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print("Data concatenation complete.")

    # --- [优化] 移除了原本的 Step 1.5 全局备份，节省大量 RAM ---
    
    # 收集所有层信息
    all_layers_subset = {}
    layers = model.model.layers
    for i in range(len(layers)):
        all_layers_subset.update({f'model.layers.{i}.{name}': layer for name, layer in find_layers(layers[i]).items()})

    # 移除 1.0 (因为背景就是 1.0，测 1.0 等于测原始模型，意义不大且浪费时间)
    candidates = sorted(list(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
    param_ratio_candidates = [r for r in candidates if r < 1.0]
    
    sensitivity_dict = {}
    
    pbar = tqdm(total=len(all_layers_subset) * len(param_ratio_candidates), desc="Calculating Sensitivity PPL")

    for full_name, layer_module in all_layers_subset.items():
        if not hasattr(layer_module, 'weight'): 
            pbar.update(len(param_ratio_candidates)); continue

        target_device = layer_module.weight.device
        sensitivity_dict[full_name] = {}
        
        i, name = int(full_name.split('.')[2]), full_name.split('.')[-2] + '.' + full_name.split('.')[-1]
        
        # 检查是否在 Hessian 中，不在则跳过
        if i not in hessian_mat or name not in hessian_mat[i]:
            for pr in param_ratio_candidates: sensitivity_dict[full_name][pr] = float('nan')
            pbar.update(len(param_ratio_candidates)); continue

        # --- [关键优化] 局部备份：只备份当前这一层 ---
        # 这一步确保我们在后面恢复时有原始数据
        W_orig_cpu = layer_module.weight.data.clone().cpu() 
        dtype = W_orig_cpu.dtype

        # 预先计算 SVD 并将组件移动到 CPU
        raw_scaling_diag_matrix = hessian_mat[i][name]
        scale_float = 'Qwen3' in args.model
        
        # 将原始权重临时移回 GPU 进行 SVD 分解
        W_orig_gpu_temp = W_orig_cpu.to(target_device)
        U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu = hessian_weight_svd(W_orig_gpu_temp.float(), raw_scaling_diag_matrix, target_device, scale_float)
        
        # SVD 结果移到 CPU
        U_cpu = U_gpu.cpu()
        S_cpu = S_gpu.cpu()
        VT_cpu = VT_gpu.cpu()
        scaling_matrix_inv_cpu = scaling_matrix_inv_gpu.cpu()
        
        del W_orig_gpu_temp, U_gpu, S_gpu, VT_gpu, scaling_matrix_inv_gpu
        torch.cuda.empty_cache()

        rows = W_orig_cpu.shape[0]
        cols = W_orig_cpu.shape[1]

        for param_ratio in param_ratio_candidates:
            # 计算保留数量
            num_s = int(rows * cols * param_ratio / (rows + cols))
            if num_s < 1: num_s = 1
            
            # CPU 切片
            truc_s_cpu = S_cpu[:num_s]
            truc_u_cpu = U_cpu[:, :num_s]
            
            # 移动切片到 GPU
            truc_s = truc_s_cpu.to(target_device)
            truc_u = truc_u_cpu.to(target_device)
            truc_vt_slice = VT_cpu[:num_s, :].to(target_device)
            scaling_inv = scaling_matrix_inv_cpu.to(target_device)
            
            # GPU 重构
            truc_v = torch.matmul(truc_vt_slice, scaling_inv)
            truc_sigma = torch.diag(truc_s)
            sqrtSigma = torch.sqrt(truc_sigma)
            svd_u = torch.matmul(truc_u, sqrtSigma)
            svd_v = torch.matmul(sqrtSigma, truc_v)
            
            W_pruned_new_target_device = (svd_u @ svd_v).to(device=target_device, dtype=dtype)
            
            # 清理中间变量
            del truc_s, truc_u, truc_vt_slice, scaling_inv, truc_v, truc_sigma, sqrtSigma, svd_u, svd_v
            
            # --- 修改模型权重 ---
            layer_module.weight.data = W_pruned_new_target_device
            
            # --- 评测 PPL ---
            # 此时：Current Layer = Pruned, Others = Original
            ppl = evaluate_perplexity(model, input_ids, args.whitening_nsamples)
            sensitivity_dict[full_name][param_ratio] = ppl
            
            pbar.update(1)
            
            del W_pruned_new_target_device
            torch.cuda.empty_cache()

        # 清理 CPU SVD 变量
        del U_cpu, S_cpu, VT_cpu, scaling_matrix_inv_cpu

        # --- [关键优化] 恢复当前层 ---
        # 使用局部备份将权重恢复为原始状态，保证下一层循环时背景是干净的
        layer_module.weight.data = W_orig_cpu.to(target_device)
        
        # 清理局部备份
        del W_orig_cpu
        torch.cuda.empty_cache()
        
    pbar.close()

    # --- 最终检查 ---
    # 理论上循环结束时，所有层都已经恢复了，这里不需要再做额外的恢复操作
    print("Sensitivity analysis complete.")
    
    # --- 保存 ---
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving sensitivity results to cache file: {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    del input_ids, all_layers_subset
    torch.cuda.empty_cache()
    gc.collect()
    
    return sensitivity_dict



@torch.no_grad()
def calib_sensitivity_ppl_dynamic(model, calib_loader, args):
    """
    双显卡优化的动态敏感度分析函数
    
    修改说明:
    1. Phase 1: 移除 n_samples_hessian 限制，使用 calib_loader 的全部数据进行 Hessian 计算和动态更新。
    2. Phase 2: 仅截取前 32 个样本进行 PPL 评估。
    3. 架构: model_dev (推理/PPL) + compute_dev (SVD分解/重构)。
    """
    
    # --- 设备设置 ---
    model_dev = args.DEV  # 主卡 (负责模型驻留、Forward、PPL)
    
    # 自动检测双卡
    if torch.cuda.device_count() > 1:
        compute_dev = torch.device('cuda:1')
    else:
        print("Warning: Single GPU detected. Fallback SVD computation to Model Device.")
        compute_dev = model_dev
        
    print(f"Device Map -> Model: {model_dev} | SVD Compute: {compute_dev}")

    model = model.cpu()
    
    # --- 准备工作 ---
    # Phase 2 评估用的样本数
    n_samples_ppl = 32 
    model_id = model.config._name_or_path.split('/')[-1]
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_ppl_dynamic_ratio_{args.ratio}_{n_samples_ppl}_{args.dataset}_seed{args.seed}.pt"
    
    if os.path.exists(cache_file):
        print(f"Loading cached sensitivity results from: {cache_file}")
        return torch.load(cache_file, map_location="cpu")

    model.eval()
    base_prune_ratio = args.ratio
    


    # 1. 初始化动态输入 (Model Device)
    print("Initializing dynamic inputs on Model Device...")
    if "opt" in model.config.model_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(model_dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(model_dev)
        if hasattr(model.model.decoder, "embed_positions"):
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(model_dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(model_dev)
        model.model.norm = model.model.norm.to(model_dev)

    # 准备 Phase 2 用的全量数据 (稍后切片)
    input_ids_all = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    
    dtype = next(iter(model.parameters())).dtype
    
    # [修改] inps 的大小直接由 loader 决定
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=model_dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # [修改] 不再检查 n_samples_hessian，只要 cache['i'] 不越界就捕获
            if cache['i'] < inps.shape[0]:
                inps[cache['i']] = inp
                cache['i'] += 1
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = kwargs.get('attention_mask')
                    cache['position_ids'] = kwargs.get('position_ids')
                    cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError 

    layers[0] = layers[0].to(model_dev)
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            model(**{k: v.to(model_dev) for k, v in batch.items()})
        except ValueError: pass
    
    layers[0] = layers[0].module.cpu()
    torch.cuda.empty_cache()

    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    
    phase1_storage = {}
    
    # ========================================================================
    # Phase 1: 构建 Base State 并收集 Hessian (使用全部 loader 数据)
    # ========================================================================
    print(f"Phase 1: Compressing to base ratio {base_prune_ratio} using all calibration data...")
    
    for i in tqdm(range(len(layers)), desc="Phase 1 (Build Base)"):
        layer = layers[i].to(model_dev) # Layer 在 Model Device
        subset = find_layers(layer)
        
        # --- A. Profile: 计算动态 Hessian (Model Device) ---
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2: inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            module.scaling_diag_matrix += torch.sum(adds, dim=0)

        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
            
        # [修改] 循环次数由 inps 实际行数决定
        num_samples = inps.shape[0]
        
        for j in range(num_samples):
            args_forward = {}
            if attention_masks is not None:
                args_forward['attention_mask'] = attention_masks[j].unsqueeze(0)
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            
            layer(inps[j].unsqueeze(0), **args_forward)
            
        for h in handles: h.remove()
        
        # --- B. Compress & Backup (Cross-Device SVD) ---
        for name in subset:
            full_name = f"model.layers.{i}.{name}"
            layer_module = subset[name]
            
            # 1. 获取 Hessian (Model Device) 并备份
            H_raw = layer_module.scaling_diag_matrix
            layer_module.scaling_diag_matrix = None
            W_orig_cpu = layer_module.weight.data.clone().cpu()
            
            # 存储 Phase 1 数据 (CPU)
            phase1_storage[full_name] = {'W_orig': W_orig_cpu, 'H': H_raw.cpu()}
            
            # 2. [跨设备] 将数据移至 Compute Device 进行 SVD
            W_compute = layer_module.weight.data.to(compute_dev).float()
            H_compute = H_raw.to(compute_dev).float()
            
            rows, cols = W_compute.shape
            num_s_base = int(rows * cols * base_prune_ratio / (rows + cols))
            scale_float = 'Qwen3' in args.model
            
            # SVD 在 GPU 1 执行
            U, S, VT, scaling_inv = hessian_weight_svd(W_compute, H_compute, compute_dev, scale_float)
            
            # 切片与重构 在 GPU 1 执行
            truc_s = S[:num_s_base]
            truc_u = U[:, :num_s_base]
            truc_v = torch.matmul(VT[:num_s_base, :], scaling_inv.float())
            sqrt_s = torch.sqrt(torch.diag(truc_s))
            W_base_compute = (truc_u @ sqrt_s) @ (sqrt_s @ truc_v)
            
            # 3. [跨设备] 仅将结果移回 Model Device
            layer_module.weight.data = W_base_compute.to(model_dev).to(dtype)
            
            # 清理 Compute Device 显存
            del W_compute, H_compute, U, S, VT, scaling_inv, truc_s, truc_u, truc_v, sqrt_s, W_base_compute
            # 清理 Model Device 显存 (H_raw)
            del H_raw 

        # --- C. Update Inputs (Model Device) ---
        for j in range(num_samples):
            args_forward = {}
            if attention_masks is not None:
                args_forward['attention_mask'] = attention_masks[j].unsqueeze(0)
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            
            inps[j] = layer(inps[j].unsqueeze(0), **args_forward)[0]
        
        layers[i] = layer.cpu() 
        torch.cuda.empty_cache() # 清理 Model Device
        with torch.cuda.device(compute_dev):
            torch.cuda.empty_cache() # 清理 Compute Device

    del inps
    gc.collect()
    torch.cuda.empty_cache()

    # 清理 Embeddings (Phase 1 结束)
    model.cpu() 
    torch.cuda.empty_cache()

    H_storage = {}
    for layer_name in phase1_storage:
        H_storage[layer_name] = phase1_storage[layer_name]['H']
    torch.save(H_storage, f"compressed_models/{model_id.replace('/','_')}_hessian_ratio{args.ratio}_{args.dataset}_{args.whitening_nsamples}_{args.seed}.pt")
    print('storage hessian matrices finished')
    # ========================================================================
    # Phase 2: Sensitivity Testing (Dual GPU)
    # ========================================================================
    print(f"Phase 2: Sensitivity Testing (Model on {model_dev}, SVD on {compute_dev})...")
    
    # 1. Model 全量上 Model Device
    model.to(model_dev)
    
    # 2. [修改] 准备 PPL 数据 (仅取前 32 个样本)
    input_ids_ppl = input_ids_all[:n_samples_ppl].to(model_dev)
    print(f"Evaluating PPL using {input_ids_ppl.shape[0]} samples.")
    
    print("Calculating Base PPL...")
    ppl_base = evaluate_perplexity(model, input_ids_ppl, n_samples_ppl)
    print(f"Base PPL (Ratio {base_prune_ratio}): {ppl_base:.4f}")

    sensitivity_dict = {}
    candidates = sorted(list(set([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, base_prune_ratio])))
    
    pbar = tqdm(total=len(phase1_storage) * len(candidates), desc="Testing Ratios")

    num_layers = len(model.model.decoder.layers) if "opt" in model.config.model_type else len(model.model.layers)

    for i in range(num_layers):
        if "opt" in model.config.model_type:
            layer = model.model.decoder.layers[i]
        else:
            layer = model.model.layers[i]
            
        subset = find_layers(layer)
        
        for name, layer_module in subset.items():
            full_name = f"model.layers.{i}.{name}"
            if full_name not in phase1_storage:
                continue
            
            sensitivity_dict[full_name] = {}
            
            data_pack = phase1_storage[full_name]
            W_orig_cpu = data_pack['W_orig']
            H_cpu = data_pack['H']
            
            # 备份 Model Device 上的 Base 权重
            W_base_backup = layer_module.weight.data.clone().detach()
            
            # --- [优化] SVD 计算 (Compute Device) ---
            # 将数据搬运到 GPU 1
            W_orig_compute = W_orig_cpu.to(compute_dev).float()
            H_compute = H_cpu.to(compute_dev).float()
            scale_float = 'Qwen3' in args.model
            
            # SVD 分解 (GPU 1)
            U, S, VT, scaling_inv = hessian_weight_svd(W_orig_compute, H_compute, compute_dev, scale_float)
            
            # 保留分解结果在 Compute Device 上，供循环使用
            del W_orig_compute, H_compute
            
            rows, cols = W_orig_cpu.shape
            
            for ratio in candidates:
                if ratio == base_prune_ratio:
                    sensitivity_dict[full_name][ratio] = ppl_base
                    pbar.update(1)
                    continue
                
                # --- 重构 (Compute Device) ---
                if ratio == 1.0:
                    # 1.0 直接从 CPU 搬运原始权重给 Model Device (最快)
                    layer_module.weight.data = W_orig_cpu.to(model_dev).to(dtype)
                else:
                    num_s = int(rows * cols * ratio / (rows + cols))
                    if num_s < 1: num_s = 1
                    
                    # 切片 (在 GPU 1 上)
                    t_s = S[:num_s]
                    t_u = U[:, :num_s]
                    t_vt = VT[:num_s, :]
                    
                    # 重构计算 (GPU 1)
                    t_v = t_vt @ scaling_inv
                    sqrt_s = torch.sqrt(torch.diag(t_s))
                    W_recon_compute = (t_u @ sqrt_s) @ (sqrt_s @ t_v)
                    
                    # [关键数据传输] GPU 1 -> GPU 0
                    layer_module.weight.data = W_recon_compute.to(model_dev).to(dtype)
                    
                    del t_s, t_u, t_vt, t_v, sqrt_s, W_recon_compute

                # --- 评估 (Model Device) ---
                # PPL 评估只使用切片后的 32 个样本
                ppl = evaluate_perplexity(model, input_ids_ppl, n_samples_ppl)
                sensitivity_dict[full_name][ratio] = ppl
                
                pbar.update(1)
            
            # 恢复权重 (Model Device)
            layer_module.weight.data = W_base_backup
            del W_base_backup, U, S, VT, scaling_inv
            
            with torch.cuda.device(compute_dev):
                torch.cuda.empty_cache()

    pbar.close()
    
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    print(f"Saving dynamic sensitivity results to: {cache_file}")
    torch.save(sensitivity_dict, cache_file)
    
    return sensitivity_dict