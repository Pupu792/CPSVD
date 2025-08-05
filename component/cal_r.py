import torch
import math


def cal_w_svd(W, H, double=False):
    if double:
        W = W.double()
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
        Scale_inv =  torch.linalg.inv(Scale)
    except Exception as e:
        print("Warning: scaling_diag_matrix is not full rank!")
        Scale += 1e-6 * torch.eye(Scale.shape[0]).to(Scale.device)
        Scale_inv =  torch.linalg.inv(Scale)

    W_scale = torch.matmul(W, Scale)
    U, S, VT = torch.linalg.svd(W_scale, full_matrices=False)

    sqrt_Sigma = torch.diag(S.sqrt()) 
    W_u = U @ sqrt_Sigma  
    W_v = sqrt_Sigma @ VT @ Scale_inv   
    # W_prime = W_u @ W_v

    return W_u.float(), W_v.float(), S.float()

def cal_svd_s(W, H, double=False):
    if double:
        W = W.double()
        H = H.double()
    Scale = torch.linalg.cholesky(H)
    # Scale_inv =  torch.linalg.inv(Scale)

    W_scale = torch.matmul(W, Scale)
    S = torch.linalg.svdvals(W_scale)

    return S

def cal_w_svd_v2(W, H, double=False):
    if double:
        H = H.double()
        W = W.double()
    Uh, Sh, VhT = torch.linalg.svd(H, full_matrices=False)
    Sx = torch.diag(Sh.sqrt())
    Wh = W@Uh@Sx
    Uwh, Swh, VwhT = torch.linalg.svd(Wh)

    # truc_u = Uwh[:, :rank]
    # truc_s = Swh[:rank]
    # truc_v = VwhT[:rank,:]
    Sx_inv = torch.diag(Sh.sqrt().reciprocal())
    Uh_inv = Uh.T
    sqrt_Sigma = torch.diag(Swh.sqrt()) 
    W_u = Uwh @ sqrt_Sigma 
    W_v = sqrt_Sigma @ VwhT @ Sx_inv @ Uh_inv
    return W_u.float(), W_v.float(), Swh.float()

def cal_svd_s_v2(W, H, double=False):
    if double:
        H = H.double()
        W = W.double()
    Uh, Sh, VhT = torch.linalg.svd(H, full_matrices=False)
    Sx = torch.diag(Sh.sqrt())
    Wh = W@Uh@Sx
    S = torch.linalg.svdvals(Wh)

    return S

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

    error_sigma = torch.sum(S_c[rank-delta_r:]**2)
    
    return error_sigma

def golden_section_search_for_minimum(indices: list, rank: int, W: torch.tensor, H: torch.tensor):
    """
    使用黄金分割搜索寻找单峰函数在离散定义域上的近似最小值。
    该方法可复用计算结果，比三元搜索更高效。
    """
    low = 0
    high = rank
    call_count = 0

    trace = {}
    
    # 黄金分割率的倒数
    inv_phi = (math.sqrt(5) - 1) / 2  # approx 0.618
    inv_phi_sq = inv_phi**2           # approx 0.382

    # 计算初始的两个内点
    # 使用 round 确保我们得到有效的离散索引
    h = high - low
    c = low + round(inv_phi_sq * h)
    
    d = low + round(inv_phi * h)
    
    yc = cal_sigma_error(W, H, indices, rank, c)
    trace[c] = yc
    yd = cal_sigma_error(W, H, indices, rank, d)
    call_count += 2
    trace[d] = yd

    while high - low >= 3:
        if yc < yd:
            # 最小值在 [low, d] 区间内
            high = d
            # 关键：旧的c成为新的d，无需重新计算d的值
            d = c
            yd = yc 
            # 只需要计算一个新的c点
            h = high - low
            c = low + round(inv_phi_sq * h)
            yc = cal_sigma_error(W, H, indices, rank, c)
            trace[c] = yc
            call_count += 1
        else:
            # 最小值在 [c, high] 区间内
            low = c
            # 关键：旧的d成为新的c，无需重新计算c的值
            c = d
            yc = yd
            # 只需要计算一个新的d点
            h = high - low
            d = low + round(inv_phi * h)
            yd = cal_sigma_error(W, H, indices, rank, d)
            trace[d] = yd
            call_count += 1
            
    # 区间足够小时，进行最终的暴力搜索
    min_y = float('inf')
    min_x = None
    # print("\n--- 最终小范围搜索 ---")
    # 确保最终检查覆盖了yc和yd
    if yc < min_y: min_y, min_x = yc, c
    if yd < min_y: min_y, min_x = yd, d
    
    for i in range(low, high + 1):
        # 避免重复计算已经算过的点
        if i == c or i == d: continue
        y = cal_sigma_error(W, H, indices, rank, i)
        trace[i] = y
        call_count += 1
        if y < min_y:
            min_y = y
            min_x = i
    
    return min_y, min_x, call_count, trace


def rank_loop(W, H, ratio):
    rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    wu, wv, ws = cal_w_svd(W, H)
    truc_wu = wu[:, :rank]
    truc_wv = wv[:rank, :]
    del_s = ws[rank:]
    error_sigma = torch.sum(del_s**2)
    truc_w = truc_wu @ truc_wv
    res_w = W - truc_w
    error_svd = torch.trace(res_w@H@res_w.T)
    
    h_diag = torch.diag(H)

    norm_res_w = torch.norm(res_w, dim=0)
    imp = norm_res_w * h_diag
    sorted_imp, indices = torch.sort(imp, descending=True)
    indices = indices.tolist()

    trace_dict = {}
    # trace_dict[0] = error_svd
    # min_error, delta_rank, call_count, trace = golden_section_search_for_minimum(indices, rank, W, H)
    for c in range(0, rank, 10):
        error = cal_sigma_error(W, H, indices, rank, c)
        trace_dict[c] = error

    return trace_dict    
    

def direct_svd(W, H, ratio):
    rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    wu, wv, ws = cal_w_svd(W, H)
    truc_wu = wu[:, :rank]
    truc_wv = wv[:rank, :]
    del_s = ws[rank:]
    # error_sigma = torch.sum(del_s**2)
    truc_w = truc_wu @ truc_wv
    return truc_w


def seg_W_SVD(W, H, ratio):

    rank = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
    wu, wv, ws = cal_w_svd(W, H)
    truc_wu = wu[:, :rank]
    truc_wv = wv[:rank, :]
    del_s = ws[rank:]
    error_sigma = torch.sum(del_s**2)
    truc_w = truc_wu @ truc_wv
    res_w = W - truc_w
    error_svd = torch.trace(res_w@H@res_w.T)
    
    h_diag = torch.diag(H)

    norm_res_w = torch.norm(res_w, dim=0)
    imp = norm_res_w * h_diag
    sorted_imp, indices = torch.sort(imp, descending=True)
    indices = indices.tolist()
    # start =time.perf_counter()
    min_error, delta_rank, call_count, _ = golden_section_search_for_minimum(indices, rank, W, H)
    
    # end = time.perf_counter()
    # print(f'search time: {end-start:.2f} 秒')

    thresh =  (error_sigma - min_error)/error_sigma
    if error_sigma > min_error:
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
    wsu, wsv, wss = cal_w_svd(W_s, H_s)
    truc_wu = wsu[:, :rank_new]
    truc_wv = wsv[:rank_new, :]
    col_indices = sorted(col_indices)
    col_w = W[:,col_indices]
    del_s = wss[rank_new:]
    error_sigma_p = torch.sum(del_s**2)
    
    # truc_wu, truc_wv = truc_wu.to(torch.float16), truc_wv.to(torch.float16)
    truc_w = truc_wu @ truc_wv
    res_w = W_s - truc_w
    error_svd_p = torch.trace(res_w@H_s@res_w.T)
    
    delta_sigma_error = (error_sigma - error_sigma_p)/error_sigma
    delta_svd_error = (error_svd - error_svd_p)/error_svd

    return truc_wu, truc_wv, col_w, rank_new, col_indices, svd_indices, delta_sigma_error, delta_svd_error