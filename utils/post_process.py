import torch
import torch.nn as nn
from tqdm import tqdm
from model_utils import find_layers
from svd_utils import hessian_weight_svd


@torch.no_grad()
def get_layer0_channel_stats(model_name, model, calib_loader, dev):
    """
    计算第0层(Layer 0)所有线性层在 *每个* 校准样本下的通道级 L2 均值 (RMS)。
    
    返回字典: 
    {
        模块名: Tensor(num_samples, hidden_size)
    }
    """
    # -------------------------------------------------------------------------
    # 1. 模型准备与 Embedding 层处理 (保持原逻辑，收集 Embedding 输出)
    # -------------------------------------------------------------------------
    print("Preparing model and collecting embedding outputs...")
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    # 存储 Embedding 层的输出，作为 Layer 0 的输入
    inps = torch.zeros(
        (len(calib_loader), 1024, model.config.hidden_size), dtype=dtype, device=dev
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
                cache['position_embeddings'] = kwargs.get('position_embeddings', None)
            else:
                cache['attention_mask'] = torch.cat(
                    (cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if kwargs.get('position_ids', None) is not None:
                    cache['position_ids'] = torch.cat(
                        (cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError

    # 使用 Catcher 替换 Layer 0
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass

    # 还原 Layer 0 和 Embedding 层
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

    # -------------------------------------------------------------------------
    # 2. 统计 Layer 0 内部线性层的通道 L2 均值
    # -------------------------------------------------------------------------
    print("Calculating channel L2 means (RMS) for Layer 0 across all samples...")
    
    layer = layers[0].to(dev)
    subset = find_layers(layer)
    
    # 临时列表存储: {模块名: [Tensor(hidden), Tensor(hidden), ...]}
    results_list = {name: [] for name in subset}

    def get_stats_hook(name):
        def hook(module, input, output):
            # input[0] 是输入张量 X, 形状可能是 [1, seq, hidden] 或 [seq, hidden]
            inp = input[0].detach()
            
            # 展平成 [TotalTokens, HiddenSize]
            inp = inp.view(-1, inp.shape[-1])
            
            # 计算每个通道的 L2 均值 (RMS): sqrt(mean(x^2))
            # dim=0 表示沿着 token 维度聚合，保留 hidden 维度
            channel_rms = torch.sqrt((inp ** 2).mean(dim=0))
            
            # 存入列表 (转到CPU)
            results_list[name].append(channel_rms.cpu())
        return hook

    # 注册 Hooks
    handles = []
    for name in subset:
        handles.append(subset[name].register_forward_hook(get_stats_hook(name)))

    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    position_embeddings = cache['position_embeddings']
    
    # 遍历所有样本
    for j in tqdm(range(inps.shape[0]), desc="Processing Samples"):
        forward_kwargs = {}
        if attention_masks is not None:
            forward_kwargs['attention_mask'] = attention_masks[j].unsqueeze(0)
        if position_ids is not None:
            forward_kwargs['position_ids'] = position_ids[0].unsqueeze(0)
        if position_embeddings is not None:
            forward_kwargs['position_embeddings'] = position_embeddings

        # 前向传播 (inps[j] 增加 batch 维度)
        layer(inps[j].unsqueeze(0), **forward_kwargs)

    # 清理 Hook
    for h in handles:
        h.remove()
    
    layer = layer.cpu()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 3. 整理结果格式
    # -------------------------------------------------------------------------
    # 将 list of tensors 转换为 tensor stacked: [num_samples, hidden_size]
    final_stats = {}
    for name, tensor_list in results_list.items():
        if len(tensor_list) > 0:
            final_stats[name] = torch.stack(tensor_list) # Shape: (N_samples, Hidden)
        else:
            final_stats[name] = None
            
    return final_stats

@torch.no_grad()
def get_layer0_svd_residuals(model_name, model, hessian_mat, ratio, dev, args):
    """
    仅针对第0层：
    1. 按照统一压缩率 (ratio) 进行 SVD 分解。
    2. 重构近似权重 W_recon。
    3. 计算残差 W_residual = W_original - W_recon。
    """
    model.eval()
    
    # --- 1. 定位第0层 ---
    if 'opt' in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers
    
    # 只取第0层
    layer_index = 0
    layer = layers[layer_index]
    
    # 查找该层内的线性模块
    subset = find_layers(layer)
    
    residuals = {}

    print(f"正在计算第 {layer_index} 层在压缩率 {ratio} 下的权重残差...")

    for name in tqdm(subset, desc="Layer 0 Modules"):
        original_layer = subset[name]
        W = original_layer.weight.data.float().to(dev)
        rows, cols = W.shape

        # --- 2. 计算目标秩 (Uniform Strategy) ---
        # Param_compressed = rank * (rows + cols)
        # Param_original = rows * cols
        # rank = (rows * cols * ratio) / (rows + cols)
        num_s_after_trunc = int(rows * cols * ratio / (rows + cols))
        
        # 确保秩至少为1，且不超过原始维度
        num_s_after_trunc = max(1, min(num_s_after_trunc, min(rows, cols)))

        # --- 3. SVD分解 (Whitening) ---
        # 注意：这里直接取 hessian_mat[0]
        if layer_index not in hessian_mat or name not in hessian_mat[layer_index]:
            print(f"警告: 缺少 {name} 的Hessian信息，跳过。")
            continue
            
        raw_scaling_diag_matrix = hessian_mat[layer_index][name].to(dev)

        scale_float = True if 'Qwen3' in args.model else False
        
        # 调用原有的分解函数
        U, S, VT, scaling_matrix_inv = hessian_weight_svd(W, raw_scaling_diag_matrix, dev, scale_float)

        # --- 4. 截断与重构 ---
        truc_s = S[:num_s_after_trunc]
        truc_u = U[:, :num_s_after_trunc]
        
        # 还原 V: 需要乘回 scaling_matrix_inv
        # W = U * S * V^T * Scale_inv
        # 这里 truc_v 计算的是 (V^T * Scale_inv) 的截断部分
        truc_v_scaled = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv.float())
        
        # 重构近似权重 W_recon = U_trunc * S_trunc * (V_trunc^T * Scale_inv)
        # 也就是 truc_u @ diag(truc_s) @ truc_v_scaled
        W_recon = torch.matmul(truc_u, torch.matmul(torch.diag(truc_s), truc_v_scaled))

        # --- 5. 计算残差 ---
        # Residual = W_original - W_reconstructed
        W_residual = W - W_recon
        
        # 存入结果字典 (转为CPU以节省显存)
        residuals[name] = W_residual.cpu()

        # --- 6. 清理内存 ---
        del W, U, S, VT, scaling_matrix_inv, truc_s, truc_u, truc_v_scaled, W_recon, W_residual, raw_scaling_diag_matrix
        torch.cuda.empty_cache()

    return residuals

def compute_weighted_channel_errors(final_stats, residuals):
    """
    计算加权的通道误差指标。
    
    Args:
        final_stats (dict): {模块名: Tensor(sample_count, in_channels)}, 记录了每个样本的激活通道强度(RMS/L2)。
        residuals (dict): {模块名: Tensor(out_channels, in_channels)}, 记录了权重分解前后的残差矩阵。
        
    Returns:
        weighted_errors (dict): {模块名: Tensor(sample_count, in_channels)}
    """
    weighted_errors = {}
    
    # 找出两个字典中共同存在的模块名
    common_modules = set(final_stats.keys()) & set(residuals.keys())
    
    print(f"Computing weighted errors for {len(common_modules)} modules...")
    
    for name in common_modules:
        # 1. 获取激活统计量 A: [sample_count, in_channels]
        # 之前计算的是 RMS (均方根)，代表激活值的平均幅度
        act_stats = final_stats[name]
        
        # 2. 获取权重残差 W_res: [out_channels, in_channels]
        res_weight = residuals[name]
        
        # 3. 计算残差的列向量 L2 范数
        # dim=0 表示沿着输出通道维度求和，得到每个输入通道对应的权重向量长度
        # Shape: [out, in] -> [in]
        # 注意: 确保在同一设备上计算，通常为了节省显存，这些都在 CPU 上
        res_col_norms = torch.norm(res_weight.float(), p=2, dim=0)
        
        # 确保设备一致 (以 act_stats 为准)
        if res_col_norms.device != act_stats.device:
            res_col_norms = res_col_norms.to(act_stats.device)
            
        # 4. 逐元素相乘
        # (Sample, In) * (In,) -> (Sample, In) 利用广播机制
        # 物理含义: 激活强度 * 权重误差强度 =该通道对输出误差的近似贡献
        weighted_error_tensor = act_stats * res_col_norms
        
        weighted_errors[name] = weighted_error_tensor
        
    return weighted_errors

