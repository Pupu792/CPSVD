import torch
import os
from tqdm import tqdm

from model_utils import find_layers
from svd_utils import hessian_weight_svd

import torch
from tqdm import tqdm

def determine_optimal_ranks_greedy(model, sensitivity_dict, target_ratio, args):
    """
    Determines the optimal rank for each module based on pre-computed PPL sensitivity data,
    using a greedy binary search approach to meet a target compression ratio.
    
    This corrected version works with the sparse PPL data (e.g., for ratios 0.1, 0.2, etc.)
    and uses a min_ratio filter instead of min_rank/rank_step.

    Args:
        model: The model to be pruned.
        sensitivity_dict (dict): The dictionary returned by calib_sensitivity_ppl.
                                 Format: {layer_name: {param_ratio: ppl, ...}}
        target_ratio (float): The target overall compression ratio for the model.
        args: An object containing configuration like min_ratio.

    Returns:
        dict: A dictionary mapping module names to their determined optimal ranks.
    """
    print("Starting optimal rank determination from PPL sensitivity data...")
    # --- MODIFICATION: Use min_ratio instead of min_rank and rank_step ---
    min_ratio = args.min_ratio
    loss_type = args.loss_type
    module_dict = {name: module for name, module in model.named_modules()}

    # ========================= Step 1: Generate All Possible Choices =========================
    print(f"Generating choices based on PPL data and filtering with min_ratio={min_ratio}...")
    all_choices = []
    
    for layer_name, results in tqdm(sensitivity_dict.items(), desc="Processing layers"):
        if layer_name not in module_dict:
            continue
            
        module = module_dict[layer_name]
        m, n = module.weight.shape
        params_orig = m * n
        
        loss_base = results.get(1.0, results.get(1))
        if loss_base is None:
            loss_base = 0
            # print(f"Warning: Base PPL for layer {layer_name} not found. Skipping.")
            # continue

        for param_ratio, loss_comp in results.items():
            # --- MODIFICATION: Filter by min_ratio ---
            if param_ratio >= 1.0 or param_ratio < min_ratio:
                continue

            # This calculation is still needed to find the corresponding rank for our choice
            rank = int(round((params_orig * param_ratio) / (m + n)))
            
            # --- MODIFICATION: REMOVED the rank_step and min_rank filtering logic ---
            # This is the key correction. We use all valid data points provided.

            if loss_type == 'ppl' or loss_type == 'kl':
                loss = loss_comp - loss_base 
            else:
                loss = loss_comp[loss_type] - loss_base[loss_type]

            params_comp = rank * (m + n)
            
            all_choices.append((layer_name, rank, loss, params_comp, params_orig))

    # Add the "do not compress" option for every layer. This is crucial for the algorithm.
    for layer_name in sensitivity_dict.keys():
        if layer_name in module_dict:
            params_orig = module_dict[layer_name].weight.numel()
            all_choices.append((layer_name, None, 0.0, params_orig, params_orig))

    print(f"Generated {len(all_choices)} total valid choices for compression.")

    # ========================= Step 2: Global Sort (No changes needed) =========================
    print("Sorting all choices globally by PPL increase (worst to best)...")
    sorted_choices = sorted(all_choices, key=lambda x: x[2], reverse=True)

    # ========================= Step 3: Binary Search (No changes needed) =========================
    print("Starting binary search for the optimal loss threshold...")
    low = 0
    high = len(sorted_choices) - 1

    while low < high:
        mid = (low + high) // 2
        layers_min_ratio = {name: 1.0 for name, module in module_dict.items() if isinstance(module, torch.nn.Linear)}
        
        for layername, r, loss, params_comp, params_orig in sorted_choices[mid:]:
            # We use the actual param_ratio from the compressed size for accuracy
            param_ratio = params_comp / params_orig
            layers_min_ratio[layername] = min(layers_min_ratio[layername], param_ratio)
            
        total_params = sum(module.weight.numel() for name, module in module_dict.items() if isinstance(module, torch.nn.Linear))
        compressed_params = sum(module_dict[name].weight.numel() * ratio for name, ratio in layers_min_ratio.items())
        current_ratio = compressed_params / total_params if total_params > 0 else 0

        if current_ratio > target_ratio:
            high = mid
        else:
            low = mid + 1
            
    best_mid = low
    print(f"Binary search complete. Optimal threshold index found at: {best_mid}")

    # ========================= Step 4: Finalize Ranks (No changes needed) =========================
    print("Generating final rank configuration based on the optimal threshold...")
    layers_best_choice = {
        name: (1.0, None) 
        for name, module in module_dict.items() 
        if isinstance(module, torch.nn.Linear)
    }

    for layername, r, loss, params_comp, params_orig in sorted_choices[best_mid:]:
        param_ratio = params_comp / params_orig
        current_best_ratio, _ = layers_best_choice[layername]
        
        if param_ratio < current_best_ratio:
            layers_best_choice[layername] = (param_ratio, r)

    final_ranks = {
        layername: rank 
        for layername, (ratio, rank) in layers_best_choice.items()
    }
    
    print("Optimal rank configuration determined successfully.")
    return final_ranks


def determine_optimal_ratios_from_ppl(model, sensitivity_dict, target_ratio):
    """
    根据预先计算好的PPL敏感度数据，使用全局排序和二分搜索的贪心策略，
    来确定在满足目标压缩率下的最优模块化压缩率配置。

    Args:
        sensitivity_dict (dict): 来自 calib_sensitivity_ppl 函数的输出。
                                 格式: {module_name: {ratio: ppl}}
        model: PyTorch 模型对象，用于获取各层的参数量。
        target_ratio (float): 目标模型总压缩率。

    Returns:
        dict: 一个映射，{模块完整名称: 最佳压缩率}。
    """
    print("根据PPL敏感度数据，开始使用贪心算法搜索最优压缩率...")
    
    module_dict = {name: module for name, module in model.named_modules()}
    
    # --- 1. 数据转换: 将PPL敏感度字典转换为扁平的 choices 列表 ---
    all_choices = []
    
    # 从第一个模块获取基线 PPL (所有模块的基线PPL都应相同)
    # a. 先找到一个有效的模块名
    first_module_name = next(iter(sensitivity_dict))
    # b. 从该模块的字典中获取key为1或1.0的ppl_base
    ppl_base = sensitivity_dict[first_module_name].get(1, sensitivity_dict[first_module_name].get(1.0))
    if ppl_base is None:
        raise ValueError("无法在sensitivity_dict中找到基线PPL (key为 1 或 1.0)")

    print(f"基线 PPL (Baseline PPL): {ppl_base:.4f}")

    for module_name, ratio_ppl_map in sensitivity_dict.items():
        for ratio, ppl in ratio_ppl_map.items():
            
            # 计算PPL增量作为“损失”
            loss = ppl - ppl_base
            all_choices.append((module_name, ratio, loss))

    # --- 2. 全局排序 ---
    # 按损失（PPL增量）降序排序，最敏感的（PPL增加最多的）排在最前面
    print(f"共生成 {len(all_choices)} 个压缩选项，开始全局排序...")
    sorted_choices = sorted(all_choices, key=lambda x: x[2], reverse=True)

    # --- 3. 二分搜索寻找最佳“敏感度阈值” ---
    print("开始二分搜索最佳敏感度阈值...")
    low = 0
    high = len(sorted_choices) - 1
    best_mid = high

    while low < high:
        mid = (low + high) // 2
        
        # a. 根据当前阈值(mid)确定每个模块的最低可接受压缩率
        layers_min_ratio = {name: 1.0 for name in sensitivity_dict.keys()}
        
        # 所有在 mid 之后的选择都是“可接受”的
        for module_name, ratio, loss in sorted_choices[mid:]:
            layers_min_ratio[module_name] = min(layers_min_ratio[module_name], ratio)
            
        # b. 计算当前配置下的总压缩率
        total_params = 0
        compressed_params = 0
        for layername, param_ratio in layers_min_ratio.items():
            if layername in module_dict:
                numel = module_dict[layername].weight.numel()
                total_params += numel
                compressed_params += numel * param_ratio
        
        current_ratio = compressed_params / total_params if total_params > 0 else 0

        # print(f"low={low}, mid={mid}, high={high} | 当前压缩率: {current_ratio:.4f} (目标: {target_ratio:.4f})")
        
        # c. 调整二分搜索的边界
        if current_ratio > target_ratio:
            # 未达到压缩目标，需要接受更多（更敏感）的压缩选项，放宽阈值
            high = mid
        else:
            # 已达到或超过压缩目标，可以尝试更严格的阈值（更低敏感度）
            low = mid + 1
            
    best_mid = low
    print(f"二分搜索完成，最佳阈值索引为: {best_mid}")

    # --- 4. 根据找到的最佳阈值，生成最终的压缩率配置 ---
    print("根据最佳阈值生成最终配置...")
    final_ratios = {name: 1.0 for name in sensitivity_dict.keys()}

    for module_name, ratio, loss in sorted_choices[best_mid:]:
        # 对于每个模块，我们选择在可接受范围内的最低压缩率
        final_ratios[module_name] = min(final_ratios[module_name], ratio)
        
    print("已生成各模块的最佳压缩率配置。")
    return final_ratios


import numpy as np
import copy

def apply_importance_weighting(sensitivity_dict, model, args, strategy="none"):
    """
    根据指定策略，为 sensitivity_dict 中的 loss 施加重要性权重。
    
    Args:
        sensitivity_dict (dict): 原始的敏感度字典。
        model: 模型实例，用于获取模块信息。
        args: 参数对象，需要包含 loss_type 和 ratio (用于找到 loss_base)。
        strategy (str): 权重策略 ("none", "random", "linear_decay", "linear_increase")。
        
    Returns:
        dict: 施加权重后的 new_sensitivity_dict。
    """
    
    print(f"Applying importance weighting strategy: {strategy}")
    if strategy == "none":
        return sensitivity_dict

    # --- Step 1: 获取有序的、将参与DP的模块列表 ---
    # DP的顺序（例如线性衰减）依赖于模块在模型中的实际顺序。
    # 我们假设 sensitivity_dict 本身是按模型顺序构建的（例如 Python 3.7+ 的 dict 或 OrderedDict）。
    module_dict = {name: module for name, module in model.named_modules()}
    ordered_valid_modules = []
    for name in sensitivity_dict.keys():
        # 筛选逻辑与您的DP函数保持一致
        if name in module_dict and hasattr(module_dict[name], "weight"):
            ordered_valid_modules.append(name)
            
    num_modules = len(ordered_valid_modules)
    if num_modules == 0:
        raise ValueError("No valid modules found for weighting.")

    # --- Step 2: 根据策略生成权重字典 {module_name: weight} ---
    module_importance = {}
    
    if strategy == "random":
        # 策略 1: 1 + 0.5 * N(0, 1)
        # 使用 np.maximum 确保权重至少为 0 (或一个很小的正数)，避免负损失
        weights = 1.0 + 0.5 * np.random.randn(num_modules)
        weights = np.maximum(weights, 0.01) # 权重至少为 0.01
        module_importance = dict(zip(ordered_valid_modules, weights))
        
    elif strategy == "linear_decay":
        # 策略 2: 前面层重要性高 (1.2)，后面层低 (0.8)
        weights = np.linspace(1.2, 0.8, num_modules)
        module_importance = dict(zip(ordered_valid_modules, weights))
        
    elif strategy == "linear_increase":
        # 策略 3: 前面层重要性低 (0.8)，后面层高 (1.2)
        weights = np.linspace(0.8, 1.2, num_modules)
        module_importance = dict(zip(ordered_valid_modules, weights))
        
    else:
        raise ValueError(f"Unknown importance strategy: {strategy}")

    # --- Step 3: 创建新的 sensitivity_dict 并应用权重 ---
    # 我们只对“损失增量” (loss - loss_base) 应用权重
    
    new_sensitivity_dict = copy.deepcopy(sensitivity_dict)
    loss_type = args.loss_type
    
    for layer_name, results in new_sensitivity_dict.items():
        if layer_name not in module_importance:
            continue # 跳过无效模块或未被选中的模块
            
        weight = module_importance[layer_name]
        
        # 1. 找到基准损失 (loss_base)
        # 严格遵循您原函数中的逻辑
        loss_base_val = results.get(args.ratio, 0)
        if isinstance(loss_base_val, dict):
            loss_base = loss_base_val.get(loss_type, 0)
        else:
            loss_base = loss_base_val
            
        # 2. 遍历所有压缩率选项，对“增量”施加权重
        for param_ratio, loss_comp in results.items():
            
            # 获取当前损失
            if isinstance(loss_comp, dict):
                current_loss = loss_comp.get(loss_type, 0)
            else:
                current_loss = loss_comp
                
            # 计算损失增量
            delta_loss = current_loss - loss_base
            
            # 计算加权后的新损失
            # new_loss = base_loss + weighted_delta
            new_loss = loss_base + (delta_loss * weight)
            
            # 写回 new_sensitivity_dict
            if isinstance(loss_comp, dict):
                new_sensitivity_dict[layer_name][param_ratio][loss_type] = new_loss
            else:
                new_sensitivity_dict[layer_name][param_ratio] = new_loss
                
    return new_sensitivity_dict


def determine_optimal_ranks_dp(model, sensitivity_dict, target_ratio, args):
    """
    使用基于参数单元的动态规划（weighted DP）方法确定每个模块的最优秩。
    保留原determine_optimal_ranks_dp中的参数量统计逻辑（从模型直接提取权重参数量），
    并替换动态规划求解部分为 find_optimal_ppl_config_weighted 的高效版本。

    Args:
        model: 待处理模型。
        sensitivity_dict (dict): 各模块敏感度数据 {module_name: {ratio: ppl或loss}}。
        target_ratio (float): 目标整体参数压缩率（压缩后/原始）。
        args: 参数对象，包含 min_ratio、loss_type、max_states 等。

    Returns:
        dict: {模块名: 对应的最优秩(rank)}
    """
    import math
    from tqdm import tqdm

    min_ratio = args.min_ratio
    loss_type = args.loss_type
    max_states = getattr(args, "max_states", 10000)

    # ===================== Step 1. 模块参数统计（保持原始逻辑） =====================
    print("Collecting module parameter statistics...")
    module_dict = {name: module for name, module in model.named_modules()}
    module_options = {}
    total_params_orig = 0

    all_module_names = set(sensitivity_dict.keys())
    for name in all_module_names:
        if name in module_dict and hasattr(module_dict[name], "weight"):
            module_options[name] = []
            total_params_orig += module_dict[name].weight.numel()

    print(f"Total parameters (original): {total_params_orig}")

    # ===================== Step 2. 构建PPL选项表 =====================
    for layer_name, results in tqdm(sensitivity_dict.items(), desc="Building DP choices"):
        if layer_name not in module_dict or not hasattr(module_dict[layer_name], "weight"):
            continue
        module = module_dict[layer_name]
        m, n = module.weight.shape
        params_orig = m * n
        # loss_base = results.get(1.0, results.get(1, float('inf')))
        loss_base = results.get(args.ratio, 0)

        for param_ratio, loss_comp in results.items():
            if param_ratio > 1.0 or param_ratio < min_ratio:
                continue

            if param_ratio == 1.0:
                params_comp = params_orig
            else:
                rank = int((params_orig * param_ratio) / (m + n))
                params_comp = rank * (m + n)
                if params_comp > params_orig:
                    continue

            if isinstance(loss_comp, dict):
                loss = loss_comp[loss_type] - loss_base[loss_type]
            else:
                loss = loss_comp - loss_base
            module_options[layer_name].append({
                "ratio": param_ratio,
                "params": params_comp,
                "loss": loss
            })

    modules = [n for n in module_options.keys() if module_options[n]]
    num_modules = len(modules)
    if num_modules == 0:
        raise ValueError("No valid modules found for DP optimization.")

    max_params = int(total_params_orig * target_ratio)
    print(f"Target compressed parameters: {max_params} ({target_ratio*100:.2f}%)")

    # ===================== Step 3. 动态参数单元划分 =====================
    param_unit = (total_params_orig // max_states) + 1
    max_units = total_params_orig // param_unit
    target_units = max_params // param_unit

    print(f"param_unit={param_unit}, max_units={max_units}, target_units={target_units}")

    # ===================== Step 4. 初始化DP表 =====================
    dp = [float('inf')] * (max_units + 1)
    path = [[1.0] * (max_units + 1) for _ in range(num_modules)]

    first_module = modules[0]
    first_params = module_dict[first_module].weight.numel()
    for opt in module_options[first_module]:
        retained_units = opt["params"] // param_unit
        if opt["loss"] < dp[retained_units]:
            dp[retained_units] = opt["loss"]
            path[0][retained_units] = opt["ratio"]

    # ===================== Step 5. 滚动DP =====================
    for i in tqdm(range(1, num_modules)):
        prev_dp = dp
        dp = [float('inf')] * (max_units + 1)
        name = modules[i]
        module_params = module_dict[name].weight.numel()
        opts = module_options[name]

        for prev_units in range(max_units + 1):
            if prev_dp[prev_units] == float('inf'):
                continue
            for opt in opts:
                cur_units = opt["params"] // param_unit
                new_units = prev_units + cur_units
                if new_units <= max_units:
                    new_loss = prev_dp[prev_units] + opt["loss"]
                    if new_loss < dp[new_units]:
                        dp[new_units] = new_loss
                        path[i][new_units] = opt["ratio"]

        # print(f"Processed module {i+1}/{num_modules}: {name}")

    # ===================== Step 6. 寻找最接近目标的可行解 =====================
    best_u, min_diff, best_loss = -1, float('inf'), float('inf')
    for u in range(max_units + 1):
        if dp[u] == float('inf'):
            continue
        diff = abs(u - target_units)
        if diff < min_diff or (diff == min_diff and dp[u] < best_loss):
            min_diff, best_u, best_loss = diff, u, dp[u]

    if best_u == -1:
        raise ValueError("No feasible solution found. Try loosening target ratio.")

    # ===================== Step 7. 回溯得到最优配置 =====================
    optimal_config = {}
    current_units = best_u
    for i in range(num_modules - 1, -1, -1):
        name = modules[i]
        chosen_ratio = path[i][current_units]
        optimal_config[name] = chosen_ratio
        retained_units = int(module_dict[name].weight.numel() * chosen_ratio) // param_unit
        current_units -= retained_units
        if current_units < 0:
            current_units = 0

    # ===================== Step 8. ratio -> rank =====================
    final_ranks = {}
    for name, ratio in optimal_config.items():
        if ratio == 1.0:
            final_ranks[name] = None
            continue
        m, n = module_dict[name].weight.shape
        params_orig = m * n
        rank = int(params_orig * ratio / (m + n))
        final_ranks[name] = rank

    print("Dynamic programming (weighted version with original param stats) finished.")
    return final_ranks