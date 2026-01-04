import torch
import torch.nn as nn
import gc
from tqdm import tqdm
import os

# 假设这些辅助函数和类在上下文环境中已定义
# from utils import find_layers, get_parent_module_and_name, CPSVD_Linear, hessian_weight_svd
from utils.svd_utils import find_layers, hessian_weight_svd
from component.cpsvd_linear import CPSVD_Linear, get_parent_module_and_name
from component.cal_r import seg_W_SVD
from utils.rank_search import determine_optimal_ranks_dp, determine_optimal_ranks_greedy


@torch.no_grad()
def whitening_dynamic(model_name, model, calib_loader, ratio, dev, args):
    """
    融合了 Profile 和 Whitening 的动态压缩函数。
    逐层处理：
    1. 获取上一层(已压缩)的输出作为输入。
    2. Profile: 在当前层(未压缩)上收集 Hessian。
    3. Compress: 执行 SVD 并替换为 CPSVD_Linear。
    4. Forward: 在当前层(已压缩)上计算输出，作为下一层的输入。
    """
    model.eval()
    
    # --- 1. 初始化输入 (Embeddings) ---
    # 这部分逻辑复用自 profle_svdllm_low_resource
    print("正在初始化校准数据输入...")
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    
    # 将第一层移到 GPU 以准备捕获输入
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # 初始化 inps: [num_batches, seqlen, hidden_size]
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    # 缓存 attention_mask 和 position_ids
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

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
                cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if kwargs.get('position_ids', None) is not None:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError # 强制中断，只为了捕获输入

    # 运行一次以捕获 embedding 层的输出
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    # 恢复第一层并清理
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

    # 准备循环所需的静态参数
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    layer_prefix = 'model.layers'
    
    # --- 2. 加载或准备 Rank 策略 (如果需要) ---
    # 如果策略是 cached，尝试加载。如果是 dynamic，我们将在循环中处理。
    model_id = model.config._name_or_path.split('/')[-1].replace('/', '_')
    ratio_str = f"{ratio:.2f}".replace('.', '_')
    cache_file_ranks = f"cache/{model_id}_optimal_ranks_{args.strategy}_ratio{ratio_str}.pt"
    optimal_ranks = {}

    if args.strategy == 'uniform':
        pass
    elif args.strategy == 'strs':
        from utils.strs import binary_search_truncation_rank
        strs_ratios = binary_search_truncation_rank(model, args)
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

    # --- 3. 主循环：逐层 Profile -> Compress -> Update ---
    print("开始动态逐层白化压缩...")


    for i in tqdm(range(len(layers)), desc="Processing Layers"):
        layer = layers[i].to(dev) # 将当前层移入 GPU
        subset = find_layers(layer)
        
        # === Step A: Profile (收集当前层的 Hessian) ===
        # 1. 注册 Hook
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            # 计算 X^T * X
            adds = torch.matmul(inp.transpose(1, 2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
        
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        
        # 2. 第一次前向传播：仅为了触发 Hook 计算 Hessian
        # 注意：这里使用的是上一层(已压缩)传来的 inps
        for j in range(inps.shape[0]):
            args_forward = {}
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            
            # 我们不需要保留这次的输出，因为层马上要变了
            layer(inps[j].unsqueeze(0), **args_forward)
        
        # 3. 移除 Hook 并收集 Hessian
        for h in handles:
            h.remove()
        
        # === Step B: Compression (SVD 分解与替换) ===
        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"
            original_layer = subset[name]
            
            # 获取刚刚计算出的 Hessian (X^T X)
            raw_scaling_diag_matrix = original_layer.scaling_diag_matrix
            # 清理 Hook 产生的大矩阵引用，防止显存泄漏
            original_layer.scaling_diag_matrix = None 
            
            W = original_layer.weight.data.float().to(dev)
            
            # 确定 Rank (num_s_after_trunc)
            if args.strategy == 'uniform':
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            elif args.strategy == 'strs':
                 pass 
            else:
                if full_name in optimal_ranks:
                    num_s_after_trunc = optimal_ranks[full_name]
                    if num_s_after_trunc is None:
                        print(f'skip layer{i}.{name}')
                        continue

            # 执行 SVD
            scale_float = True if 'Qwen3' in args.model else False
            U, S, VT, scaling_matrix_inv = hessian_weight_svd(W, raw_scaling_diag_matrix, dev, scale_float)
            
            truc_s = S[:num_s_after_trunc]
            truc_u = U[:, :num_s_after_trunc]
            truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv.float())
            
            sqrt_sigma = torch.diag(torch.sqrt(truc_s))
            svd_u = torch.matmul(truc_u, sqrt_sigma)
            svd_v = torch.matmul(sqrt_sigma, truc_v)
            
            # 移回 CPU 准备赋值
            svd_u = svd_u.to(original_layer.weight.dtype)
            svd_v = svd_v.to(original_layer.weight.dtype)
            
            # 创建并替换为 CPSVD_Linear
            new_svd_layer = CPSVD_Linear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=num_s_after_trunc,
                col_indices=[],
                bias=original_layer.bias is not None
            )
            new_svd_layer.v_proj.weight.data = svd_v
            new_svd_layer.u_proj.weight.data = svd_u
            if original_layer.bias is not None:
                new_svd_layer.u_proj.bias.data = original_layer.bias.data
            
            parent_module, module_name = get_parent_module_and_name(model, full_name)
            setattr(parent_module, module_name, new_svd_layer)
            
            # 清理显存
            del W, U, S, VT, truc_s, truc_u, truc_v, sqrt_sigma, svd_u, svd_v, raw_scaling_diag_matrix
            torch.cuda.empty_cache()

        # === Step C: Update (生成下一层的输入) ===
        # 此时 layer 已经是压缩后的结构了 (CPSVD_Linear)
        # 再次运行前向传播，计算出的结果将包含压缩带来的误差/偏移
        for j in range(inps.shape[0]):
            args_forward = {}
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            
            # 记录输出到 outs
            inps[j] = layer(inps[j].unsqueeze(0), **args_forward)[0]
        
        # 将当前层移回 CPU 以节省显存
        layer = layer.cpu() 
        layers[i] = layer # 确保模型中的引用更新（虽然 setattr 已经修改了子模块，但 move cpu 需要这一步）
        
        torch.cuda.empty_cache()
        gc.collect()

    print("动态白化压缩完成。")


@torch.no_grad()
def whitening_cp_dynamic(model_name, model, calib_loader, ratio, dev, args):
    """
    动态版本的 Whitening CP：
    1. 逐层捕获输入激活值并计算 Hessian。
    2. 依据 Hessian 对当前层进行 SVD + Column Prefix 压缩。
    3. 用压缩后的层生成下一层的输入。
    """
    model.eval()
    dtype = next(iter(model.parameters())).dtype

    # --- Step 1: 准备输入捕获 (Embeddings) ---
    if 'opt' in model_name:
        layers = model.model.decoder.layers
        layer_prefix = 'model.decoder.layers'
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    else:
        layers = model.model.layers
        layer_prefix = 'model.layers'
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    # 初始化 inps 用于存储各 batch 的输入
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None, 'position_embeddings': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs.get('attention_mask')
                cache['position_ids'] = kwargs.get('position_ids')
                cache['position_embeddings'] = kwargs.get('position_embeddings')
            raise ValueError # 捕获到第一层输入后中断

    layers[0] = layers[0].to(dev)
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            model(**{k: v.to(dev) for k, v in batch.items()})
        except ValueError: pass
    
    layers[0] = layers[0].module.cpu()
    torch.cuda.empty_cache()

    # --- Step 2: 确定 Ranks 策略 ---
    model_id = model.config._name_or_path.split('/')[-1].replace('/', '_')
    ratio_str = f"{ratio:.2f}".replace('.', '_') # e.g., 0.70 -> 0_70
    cache_file_ranks = f"cache/{model_id}_optimal_ranks_{args.strategy}_ratio{ratio_str}.pt"
    
    if args.strategy == 'uniform':
        pass
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
                optimal_ranks = {'q_proj':656, 'k_proj':1392, 'o_proj':2128, 'gate_proj':2128, 'up_proj':2352, 'down_proj': 2312}
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


    # --- Step 3: 逐层 Profile -> Compress -> Forward ---
    error_metrics = {}
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']

    print("开始动态逐层白化与 SVD 替换...")
    for i in tqdm(range(len(layers)), desc="Processing Layers"):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        error_metrics[f"layer_{i}"] = {}

        # A. Profile: 注册 Hook 并运行一次 Forward 收集 Hessian
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2: inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1, 2), inp)
            module.scaling_diag_matrix += torch.sum(adds, dim=0)
            del inp, adds

        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))

        # 运行 Profile 前向 (使用当前的 inps)
        for j in range(inps.shape[0]):
            args_forward = {}
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            layer(inps[j].unsqueeze(0), **args_forward)

        for h in handles: h.remove()

        # B. Compress: 对当前层进行 SVD + CPSVD 替换
        for name in subset:
            full_name = f"{layer_prefix}.{i}.{name}"
            original_layer = subset[name]
            W = original_layer.weight.data.float().to(dev)
            raw_scaling_diag_matrix = original_layer.scaling_diag_matrix
            original_layer.scaling_diag_matrix = None # 释放 Hessian 显存

            # 确定 Rank (num_s_after_trunc)
            if args.strategy == 'uniform':
                num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            else:
                # 此处根据之前的 optimal_ranks 逻辑获取 rank
                num_s_after_trunc = optimal_ranks.get(full_name, None)
                if num_s_after_trunc is None: continue

            scale_float = 'Qwen3' in args.model
            # 调用原本的计算函数
            truc_wu, truc_wv, col_w, svd_rank, keep_indices, drop_indices, sigma_err, svd_err = seg_W_SVD(
                W, raw_scaling_diag_matrix, num_s_after_trunc, scale_float
            )

            error_metrics[f"layer_{i}"][name] = {
                "sigma_error_reduction": sigma_err.item() if torch.is_tensor(sigma_err) else sigma_err,
                "svd_error_reduction": svd_err.item() if torch.is_tensor(svd_err) else svd_err
            }

            # 替换层逻辑
            new_svd_layer = CPSVD_Linear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=svd_rank,
                col_indices=keep_indices,
                bias=original_layer.bias is not None
            ).to(dtype=original_layer.weight.dtype)

            new_svd_layer.u_proj.weight.data = truc_wu.to(original_layer.weight.dtype)
            new_svd_layer.v_proj.weight.data = truc_wv.to(original_layer.weight.dtype)
            new_svd_layer.svd_indices = new_svd_layer.svd_indices.to(device=dev)
            new_svd_layer.col_indices = new_svd_layer.col_indices.to(device=dev)
            if hasattr(new_svd_layer, 'col_proj'):
                new_svd_layer.col_proj.weight.data = col_w.to(original_layer.weight.dtype)
            if original_layer.bias is not None:
                new_svd_layer.u_proj.bias.data = original_layer.bias.data

            parent_module, module_name = get_parent_module_and_name(model, full_name)
            setattr(parent_module, module_name, new_svd_layer)
            
            del W, raw_scaling_diag_matrix, truc_wu, truc_wv, col_w

        # C. Update Forward: 用替换后的层计算下一层的输入 (原地更新 inps)
        for j in range(inps.shape[0]):
            args_forward = {}
            if position_ids is not None:
                args_forward['position_ids'] = position_ids[0].unsqueeze(0)
            if position_embeddings is not None:
                args_forward['position_embeddings'] = position_embeddings
            
            # 使用压缩后的 layer 重新计算输出
            inps[j] = layer(inps[j].unsqueeze(0), **args_forward)[0]

        layers[i] = layer.cpu() # 移回 CPU
        torch.cuda.empty_cache()
        gc.collect()

    # --- Step 4: 保存 Error Metrics (保持原逻辑) ---
    # ... (此处省略 json.dump 逻辑) ...
    print("模型已完成动态白化压缩与原位替换。")