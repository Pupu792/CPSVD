import torch
import math


def cal_w_svd(W, H, scale_float):
    H = H.double()
    try:
        Scale = torch.linalg.cholesky(H)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not positive-definite!")
        percdamp = 0.05
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        Scale = torch.linalg.cholesky(H)
    try:
        if scale_float:
            Scale = Scale.float()
        Scale_inv =  torch.linalg.inv(Scale)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not full rank!")
        Scale += 1e-6 * torch.eye(Scale.shape[0]).to(Scale.device)
        Scale_inv =  torch.linalg.inv(Scale)
    
    Scale = Scale.float()
    Scale_inv = Scale_inv.float()
    W_scale = torch.matmul(W, Scale)
    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

    sqrt_Sigma = torch.diag(S.sqrt()) 
    W_u = U @ sqrt_Sigma  
    W_v = sqrt_Sigma @ VT @ Scale_inv   
    # W_prime = W_u @ W_v

    return W_u.float(), W_v.float(), S.float()

def cal_svd_s(W, H):

    H = H.double()
    try:
        Scale = torch.linalg.cholesky(H)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not positive-definite!")
        percdamp = 0.05
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        Scale = torch.linalg.cholesky(H)
    Scale = Scale.float()
    W_scale = torch.matmul(W, Scale)
    S = torch.linalg.svdvals(W_scale)

    return S

# def cal_w_svd_v2(W, H, double=False):
#     if double:
#         H = H.double()
#         W = W.double()
#     Uh, Sh, VhT = torch.linalg.svd(H, full_matrices=False)
#     Sx = torch.diag(Sh.sqrt())
#     Wh = W@Uh@Sx
#     Uwh, Swh, VwhT = torch.linalg.svd(Wh)

#     # truc_u = Uwh[:, :rank]
#     # truc_s = Swh[:rank]
#     # truc_v = VwhT[:rank,:]
#     Sx_inv = torch.diag(Sh.sqrt().reciprocal())
#     Uh_inv = Uh.T
#     sqrt_Sigma = torch.diag(Swh.sqrt()) 
#     W_u = Uwh @ sqrt_Sigma 
#     W_v = sqrt_Sigma @ VwhT @ Sx_inv @ Uh_inv
#     return W_u.float(), W_v.float(), Swh.float()

# def cal_svd_s_v2(W, H, double=False):
#     if double:
#         H = H.double()
#         W = W.double()
#     Uh, Sh, VhT = torch.linalg.svd(H, full_matrices=False)
#     Sx = torch.diag(Sh.sqrt())
#     Wh = W@Uh@Sx
#     S = torch.linalg.svdvals(Wh)

#     return S

def cal_sub_W_H(W, H, indices, rank, delta_r):
    full_indices = list(range(W.shape[1]))
    rank_c = rank - delta_r
    f = (W.shape[0]+W.shape[1])/(W.shape[0]-rank_c)
    delta_c = round(delta_r*f)
    col_indices = indices[:delta_c]
    svd_indices = [i for i in full_indices if i not in col_indices]
    W_c = W[:,svd_indices]
    H_c = H[svd_indices,:]
    H_c = H_c[:,svd_indices]
    return W_c, H_c, col_indices, svd_indices

def cal_sigma_error(W, H, indices, rank, delta_r):
    W_c, H_c, _, _ = cal_sub_W_H(W, H, indices, rank, delta_r)
    S_c = cal_svd_s(W_c, H_c)

    error_sigma = torch.sqrt(torch.sum(S_c[rank-delta_r:]**2))
    
    return error_sigma


def golden_section_search_for_minimum(indices: list, rank: int, W: torch.Tensor, H: torch.Tensor, y0: torch.Tensor):
    """
    使用改进的黄金分割搜索寻找单峰函数在离散定义域上的近似最小值。
    优化点：
    1. 初始阶段先计算一个较小x的函数值，如若明显不优则直接退出；
    2. 该较小x点结果可在后续搜索中重复使用；
    3. 搜索结束时不进行low-high范围的遍历，而直接比较边界与内点。
    """
    low = 0
    high = rank
    call_count = 0
    trace = {}
    # ---------- (1) 黄金分割搜索 ----------
    inv_phi = (math.sqrt(5) - 1) / 2      # ≈0.618
    inv_phi_sq = inv_phi ** 2              # ≈0.382

    # ---------- (2) 初始快速判断 ----------
    # 选取一个很小的x作为先验点（例如 rank * 0.05 处）
    x_small1 = max(1, round(rank * inv_phi_sq))
    y_small1 = cal_sigma_error(W, H, indices, rank, x_small1)
    trace[x_small1] = y_small1
    call_count += 1

    x_small2 = max(1, round(rank * inv_phi * inv_phi_sq**2))
    y_small2 = cal_sigma_error(W, H, indices, rank, x_small2)
    trace[x_small2] = y_small2
    call_count += 1
    # 如果小x处性能比y0更差，则直接返回y0（表示不值得进一步搜索）
    if y_small1 >= y0 and y_small2 >= y0:
        return y0, 0, call_count, trace

    h = high - low
    c = low + round(inv_phi_sq * h)
    d = low + round(inv_phi * h)

    # 复用之前计算的小x点（若刚好命中）
    if c == x_small1:
        yc = y_small1
    elif c == x_small2:
        yc = y_small2
    else:
        yc = cal_sigma_error(W, H, indices, rank, c)
        trace[c] = yc
        call_count += 1

    if d == x_small1:
        yd = y_small1
    elif d == x_small2:
        yd = y_small2
    else:
        yd = cal_sigma_error(W, H, indices, rank, d)
        trace[d] = yd
        call_count += 1

    # 黄金分割主循环
    while high - low >= 3:
        if yc < yd:
            high = d
            d = c
            yd = yc
            h = high - low
            c = low + round(inv_phi_sq * h)
            if c in trace:
                yc = trace[c]
            else:
                yc = cal_sigma_error(W, H, indices, rank, c)
                trace[c] = yc
                call_count += 1
        else:
            low = c
            c = d
            yc = yd
            h = high - low
            d = low + round(inv_phi * h)
            if d in trace:
                yd = trace[d]
            else:
                yd = cal_sigma_error(W, H, indices, rank, d)
                trace[d] = yd
                call_count += 1

    min_y = float('inf')
    min_x = None
    # print("\n--- 最终小范围搜索 ---")
    # 确保最终检查覆盖了yc和yd
    if yc < min_y: min_y, min_x = yc, c
    if yd < min_y: min_y, min_x = yd, d

    return min_y, min_x, call_count, trace


    

# def direct_svd(W, H, ratio):
#     rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
#     wu, wv, ws = cal_w_svd(W, H)
#     truc_wu = wu[:, :rank]
#     truc_wv = wv[:rank, :]
#     del_s = ws[rank:]
#     # error_sigma = torch.sum(del_s**2)
#     truc_w = truc_wu @ truc_wv
#     return truc_w


def seg_W_SVD(W, H, rank, scale_float):

    # rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    wu, wv, ws = cal_w_svd(W, H, scale_float)
    truc_wu = wu[:, :rank]
    truc_wv = wv[:rank, :]
    del_s = ws[rank:]
    error_sigma = torch.sqrt(torch.sum(del_s**2))
    truc_w = truc_wu @ truc_wv
    res_w = W - truc_w
    error_svd = torch.sqrt(torch.trace(res_w@H@res_w.T))
    
    h_diag = torch.diag(H)

    norm_res_w = torch.norm(res_w, dim=0)
    imp = norm_res_w**2 * h_diag
    sorted_imp, indices = torch.sort(imp, descending=True)
    indices = indices.tolist()
    # start =time.perf_counter()
    min_error, delta_rank, call_count, _ = golden_section_search_for_minimum(indices, rank, W, H, error_sigma)
    
    # end = time.perf_counter()
    # print(f'search time: {end-start:.2f} 秒')

    if (error_sigma - min_error)/error_sigma > 0.01:
        delta_rank = delta_rank
    else:
        delta_rank = 0
        # yN = cal_sigma_error(W, H, indices, rank, rank)
        # if error_sigma < yN:
        #     delta_rank = 0
        # else:
        #     delta_rank = rank
            
    rank_new = rank - delta_rank
    W_s, H_s, col_indices, svd_indices = cal_sub_W_H(W, H, indices, rank, delta_rank)
    wsu, wsv, wss = cal_w_svd(W_s, H_s, scale_float)
    truc_wu = wsu[:, :rank_new]
    truc_wv = wsv[:rank_new, :]
    col_indices = sorted(col_indices)
    col_w = W[:,col_indices]
    del_s = wss[rank_new:]
    error_sigma_p = torch.sqrt(torch.sum(del_s**2))
    
    # truc_wu, truc_wv = truc_wu.to(torch.float16), truc_wv.to(torch.float16)
    truc_w = truc_wu @ truc_wv
    res_w = W_s - truc_w
    error_svd_p = torch.sqrt(torch.trace(res_w@H_s@res_w.T))
    
    delta_sigma_error = (error_sigma - error_sigma_p)/error_sigma
    delta_svd_error = (error_svd - error_svd_p)/error_svd

    return truc_wu, truc_wv, col_w, rank_new, col_indices, svd_indices, delta_sigma_error, delta_svd_error


def analyze_column_loss(W, H, rank, scale_float, step=5):
    """
    分析保留列的数量与近似误差之间的关系。
    
    参数:
        W: 权重矩阵
        H: Hessian 矩阵
        rank: 目标总秩
        scale_float: 是否转换精度
        step: 扫描 delta_r 的步长（默认为1，越大越快但精度越低）
        
    返回:
        kept_columns_list: 保留的列数列表 (x轴)
        errors_list: 对应的误差值列表 (y轴)
    """
    
    # 1. --- 预处理阶段 (与 seg_W_SVD 保持一致) ---
    # 我们必须先计算 importance 来确定列的排序 indices
    wu, wv, ws = cal_w_svd(W, H, scale_float)
    truc_wu = wu[:, :rank]
    truc_wv = wv[:rank, :]
    
    # 计算初始 SVD 近似
    truc_w = truc_wu @ truc_wv
    res_w = W - truc_w
    
    # 计算重要性 (Importance)
    h_diag = torch.diag(H)
    norm_res_w = torch.norm(res_w, dim=0)
    imp = norm_res_w**2 * h_diag
    sorted_imp, indices = torch.sort(imp, descending=True)
    indices = indices.tolist()
    
    # 2. --- 扫描阶段 ---
    column_error = {}
    
    # delta_r 代表从 SVD 部分削减的秩，这部分秩被转化为保留的列
    # 范围从 0 到 rank
    # 使用 step 减少计算量，特别是当 rank 很大时
    search_range = range(0, rank + 1, step)
    
    print(f"Scanning {len(search_range)} points...")
    
    for delta_r in search_range:
        # 计算对应的误差
        # cal_sigma_error 内部会调用 cal_sub_W_H 进行分割并计算剩余奇异值误差
        current_error = cal_sigma_error(W, H, indices, rank, delta_r)
        
        # 计算实际保留的列数 (delta_c)
        # 逻辑需与 cal_sub_W_H 中的一致
        rank_c = rank - delta_r
        # 防止分母为0 (虽料理论上 rank_c 不会等于 W.shape[0]，但做个保护)
        denom = W.shape[0] - rank_c
        if denom == 0: denom = 1e-6 
        
        f = (W.shape[0] + W.shape[1]) / denom
        delta_c = round(delta_r * f)
        
        column_error[delta_c] = current_error.item()
        
    return column_error