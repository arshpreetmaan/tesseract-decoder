import os
import glob
import json
import argparse
import pathlib
import stim
import numpy as np
from scipy.sparse import csc_matrix

def get_detector_types(circuit: stim.Circuit):
    coords = circuit.get_detector_coordinates()
    n_det = circuit.num_detectors
    det_types = np.zeros(n_det, dtype=int)
    for i in range(n_det):
        if coords.get(i)[3] >=3:
            # 3 means Z detector, 1 means X detector
            det_types[i] = 3 
        else:
            det_types[i] = 1 # default
    return det_types

def dem_to_check_matrices(dem: stim.DetectorErrorModel, allow_undecomposed_hyperedges=True):
    errors = []
    observables_list = []
    priors = []
    for inst in dem.flattened():
        if inst.type == "error":
            priors.append(inst.args_copy()[0])
            targets = inst.targets_copy()
            dets = [t.val for t in targets if t.is_relative_detector_id()]
            obs = [t.val for t in targets if t.is_logical_observable_id()]
            errors.append(dets)
            observables_list.append(obs)
            
    M = dem.num_detectors
    N = len(errors)
    O = dem.num_observables
    
    row_ind = []
    col_ind = []
    data = []
    for j, dets in enumerate(errors):
        for d in dets:
            row_ind.append(d)
            col_ind.append(j)
            data.append(1)
            
    H = csc_matrix((data, (row_ind, col_ind)), shape=(M, N), dtype=np.uint8)

    obs_row_ind = []
    obs_col_ind = []
    obs_data = []
    for j, obs in enumerate(observables_list):
        for o in obs:
            obs_row_ind.append(o)
            obs_col_ind.append(j)
            obs_data.append(1)
            
    L = csc_matrix((obs_data, (obs_row_ind, obs_col_ind)), shape=(O, N), dtype=np.uint8)
    
    return H, L, np.array(priors), errors

def matrices_to_dem(H: csc_matrix, L: csc_matrix, priors: np.ndarray) -> stim.DetectorErrorModel:
    dem = stim.DetectorErrorModel()
    H_csc = H.tocsc()
    L_csc = L.tocsc()
    
    for j in range(H_csc.shape[1]):
        p = priors[j]
        targets = []
        
        # Add detectors
        for i in H_csc.indices[H_csc.indptr[j]:H_csc.indptr[j+1]]:
            targets.append(stim.target_relative_detector_id(int(i)))
            
        # Add observables
        for o in L_csc.indices[L_csc.indptr[j]:L_csc.indptr[j+1]]:
            targets.append(stim.target_logical_observable_id(int(o)))
            
        if targets:
            dem.append("error", p, targets)
            
    return dem

def gari_transform(H: csc_matrix, L: csc_matrix, det_types: np.ndarray, priors: np.ndarray, return_dem: bool = False):
    """
    Applies the Gari Transform (arXiv:2510.14060) to the check matrix.
    Splits Y errors into independent X and Z components and adds virtual detectors.
    """
    # Separate rows into X-detectors (detects Z errors) and Z-detectors (detects X errors)
    is_x_det = (det_types == 1)
    is_z_det = (det_types == 3)
    
    H_csr = H.tocsr()
    hx = H_csr[is_x_det, :]
    hz = H_csr[is_z_det, :]
    
    # Identify variables (errors)
    hx_nnz = hx.getnnz(axis=0)
    hz_nnz = hz.getnnz(axis=0)
    
    hx_any = hx_nnz > 0
    hz_any = hz_nnz > 0
    
    i_hy = np.where(hx_any & hz_any)[0]
    i_hx_only = np.where(hx_any & ~hz_any)[0]
    i_hz_only = np.where(~hx_any & hz_any)[0]
    
    hx_csc = hx.tocsc()
    hz_csc = hz.tocsc()
    L_csc = L.tocsc()
    
    dx = hx_csc[:, i_hx_only]
    dz = hz_csc[:, i_hz_only]
    hx_yonly = hx_csc[:, i_hy]
    hz_yonly = hz_csc[:, i_hy]
    
    L_dx = L_csc[:, i_hx_only]
    L_dz = L_csc[:, i_hz_only]
    L_y = L_csc[:, i_hy]
    
    P_dx = priors[i_hx_only]
    P_dz = priors[i_hz_only]
    P_y = priors[i_hy]
    
    mx, nx = dx.shape
    mz, nz = dz.shape
    ny = hx_yonly.shape[1]
    
    # Helper to find matching columns efficiently using hashing
    def get_col_hashes(mat_csc):
        hashes = []
        for j in range(mat_csc.shape[1]):
            start = mat_csc.indptr[j]
            end = mat_csc.indptr[j+1]
            hashes.append(tuple(mat_csc.indices[start:end]))
        return hashes

    dx_hashes = get_col_hashes(dx)
    dz_hashes = get_col_hashes(dz)
    
    dx_hash_to_idx = {h: i for i, h in enumerate(dx_hashes)}
    dz_hash_to_idx = {h: i for i, h in enumerate(dz_hashes)}
    
    hx_yonly_hashes = get_col_hashes(hx_yonly)
    hz_yonly_hashes = get_col_hashes(hz_yonly)
    
    U_rows, U_cols = [], []
    for j, h in enumerate(hx_yonly_hashes):
        i = dx_hash_to_idx.get(h, -1)
        if i != -1:
            U_rows.append(i)
            U_cols.append(j)
    U_data = np.ones(len(U_rows), dtype=np.uint8)
    U = csc_matrix((U_data, (U_rows, U_cols)), shape=(nx, ny), dtype=np.uint8)
    
    V_rows, V_cols = [], []
    for j, h in enumerate(hz_yonly_hashes):
        i = dz_hash_to_idx.get(h, -1)
        if i != -1:
            V_rows.append(i)
            V_cols.append(j)
    V_data = np.ones(len(V_rows), dtype=np.uint8)
    V = csc_matrix((V_data, (V_rows, V_cols)), shape=(nz, ny), dtype=np.uint8)
    
    from scipy.sparse import eye, bmat
    
    I_nx = eye(nx, format='csc', dtype=np.uint8)
    I_nz = eye(nz, format='csc', dtype=np.uint8)
    
    # Gari matrix structure:
    # [ 0      0          0             dx      0   ]
    # [ 0      0          0             0       dz  ]
    # [ I_nx   0          U             I_nx    0   ]
    # [ 0      I_nz       V             0       I_nz]
    
    blocks = [
        [None, None, None, dx,   None],
        [None, None, None, None, dz  ],
        [I_nx, None, U,    I_nx, None],
        [None, I_nz, V,    None, I_nz]
    ]
    
    gari_matrix = bmat(blocks, format='csc', dtype=np.uint8)
    
    # The columns are ordered as: [ e_z, e_x, e_y, e'_z, e'_x ]
    # This matches the blocks:
    # [None, None, None, dx,   None]  -> dx hits e'_z
    # [None, None, None, None, dz  ]  -> dz hits e'_x
    # [I_nx, None, U,    I_nx, None]  -> e'_z = e_z + U*e_y
    # [None, I_nz, V,    None, I_nz]  -> e'_x = e_x + V*e_y
    
    L_ez = L_dx.astype(np.uint8)
    L_ex = L_dz.astype(np.uint8)
    L_ey = L_y.astype(np.uint8)
    
    O_num = L.shape[0]
    Z_nx = csc_matrix((O_num, nx), dtype=np.uint8)
    Z_nz = csc_matrix((O_num, nz), dtype=np.uint8)
    Z_ny = csc_matrix((O_num, ny), dtype=np.uint8)
    
    L_blocks = [
        [Z_nx, Z_nz, Z_ny, L_dx.astype(np.uint8), L_dz.astype(np.uint8)]
    ]
    gari_obs_matrix = bmat(L_blocks, format='csc', dtype=np.uint8)
    
    L_blocks_og = [
        [L_dx.astype(np.uint8), L_dz.astype(np.uint8), L_ey, Z_nx, Z_nz]
    ]
    gari_obs_matrix_og = bmat(L_blocks_og, format='csc', dtype=np.uint8)
    
    P_ez = P_dx
    P_ex = P_dz
    P_ey = P_y
    
    # print("\n--- Physical Priors (From Stim) ---")
    # print(f"P_ez (Hidden Z): min={np.min(P_ez):.2e}, max={np.max(P_ez):.2e}, mean={np.mean(P_ez):.2e}")
    # print(f"P_ex (Hidden X): min={np.min(P_ex):.2e}, max={np.max(P_ex):.2e}, mean={np.mean(P_ex):.2e}")
    # if len(P_ey) > 0:
    #     print(f"P_ey (Hidden Y): min={np.min(P_ey):.2e}, max={np.max(P_ey):.2e}, mean={np.mean(P_ey):.2e}")
    # else:
    #     print("P_ey (Hidden Y): None")
    print("-----------------------------------")
    
    # --- Legacy Aggregated Priors ---
    P_ez_prime_agg = P_ez + U @ P_ey
    P_ex_prime_agg = P_ex + V @ P_ey
    gari_priors_agg = np.concatenate([P_ez, P_ex, P_ey, P_ez_prime_agg, P_ex_prime_agg])
    
    # --- Helper Functions ---
    def cost(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log((1-p)/p)
    def prob(c):
        c = np.clip(c, -30, 30)
        return 1 / (1 + np.exp(c))

    # --- Mode B: ez,ex,ey Keep, ez',ex' Free ---
    gari_priors_keep_free = np.concatenate([
        P_ez, P_ex, P_ey,
        np.full_like(P_ez, 0.499),
        np.full_like(P_ex, 0.499)
    ])

    # --- Mode C: ez,ex,ey Keep, ez',ex' Scaled from Agg ---
    lam = 0.001
    C_z_prime_agg = cost(P_ez_prime_agg)
    C_x_prime_agg = cost(P_ex_prime_agg)
    gari_priors_keep_scaled = np.concatenate([
        P_ez, P_ex, P_ey,
        prob(lam * C_z_prime_agg),
        prob(lam * C_x_prime_agg)
    ])

    # --- Mode D: ez,ex,ey Free, ez',ex' Penalized ---
    gari_priors_hidden_free = np.concatenate([
        np.full_like(P_ez, 0.499), 
        np.full_like(P_ex, 0.499), 
        np.full_like(P_ey, 0.499), 
        P_ez, P_ex
    ])

    # --- Mode E: ez,ex,ey Scaled, ez',ex' Penalized ---
    C_z = cost(P_ez)
    C_x = cost(P_ex)
    C_y = cost(P_ey)
    gari_priors_tiny = np.concatenate([
        prob(lam * C_z), 
        prob(lam * C_x), 
        prob(lam * C_y), 
        P_ez, P_ex
    ])
    
    # --- Mode F: ez,ex,ey Free, ez',ex' Aggregated ---
    gari_priors_hf_agg = np.concatenate([
        np.full_like(P_ez, 0.499), 
        np.full_like(P_ex, 0.499), 
        np.full_like(P_ey, 0.499), 
        P_ez_prime_agg, P_ex_prime_agg
    ])

    # --- Mode G: ez,ex,ey Scaled, ez',ex' Aggregated ---
    gari_priors_tiny_agg = np.concatenate([
        prob(lam * C_z), 
        prob(lam * C_x), 
        prob(lam * C_y), 
        P_ez_prime_agg, P_ex_prime_agg
    ])

    # --- Mode H: ez,ex,ey Keep, ez',ex' Keep ---
    gari_priors_keep_keep = np.concatenate([
        P_ez, P_ex, P_ey, 
        P_ez, P_ex
    ])

    U_weighted = U.multiply(P_ey)
    max_ey_to_z = U_weighted.max(axis=1).toarray().flatten()
    P_ez_prime_max = np.maximum(P_ez, max_ey_to_z)

    V_weighted = V.multiply(P_ey)
    max_ey_to_x = V_weighted.max(axis=1).toarray().flatten()
    P_ex_prime_max = np.maximum(P_ex, max_ey_to_x)

    # --- Mode I: ez,ex,ey Keep, ez',ex' Maxed ---
    gari_priors_keep_max = np.concatenate([
        P_ez, P_ex, P_ey, 
        P_ez_prime_max, P_ex_prime_max
    ])

    # --- Mode J: ez,ex,ey Free, ez',ex' Maxed ---
    gari_priors_hf_max = np.concatenate([
        np.full_like(P_ez, 0.499), 
        np.full_like(P_ex, 0.499), 
        np.full_like(P_ey, 0.499), 
        P_ez_prime_max, P_ex_prime_max
    ])

    # --- Mode K: ez,ex,ey Scaled, ez',ex' Maxed ---
    gari_priors_tiny_max = np.concatenate([
        prob(lam * C_z), 
        prob(lam * C_x), 
        prob(lam * C_y), 
        P_ez_prime_max, P_ex_prime_max
    ])

    # --- Mode L: ez,ex Keep, ey Free, ez',ex' Aggregated ---
    gari_priors_keep_freeY_agg = np.concatenate([
        P_ez, P_ex, np.full_like(P_ey, 0.499),
        P_ez_prime_agg, P_ex_prime_agg
    ])

    # --- Mode M: ez,ex Keep, ey Free, ez',ex' Maxed ---
    gari_priors_keep_freeY_max = np.concatenate([
        P_ez, P_ex, np.full_like(P_ey, 0.499),
        P_ez_prime_max, P_ex_prime_max
    ])
    
    # --- Mode N: ez, ex, ey Keep, ez', ex' XOR Aggregated ---
    eps = 1e-15
    log_1m2_P_ez = np.log(np.clip(1 - 2 * P_ez, eps, 1.0))
    log_1m2_P_ex = np.log(np.clip(1 - 2 * P_ex, eps, 1.0))
    log_1m2_P_ey = np.log(np.clip(1 - 2 * P_ey, eps, 1.0))
    
    P_ez_prime_xor = 0.5 * (1 - np.exp(log_1m2_P_ez + U @ log_1m2_P_ey))
    P_ex_prime_xor = 0.5 * (1 - np.exp(log_1m2_P_ex + V @ log_1m2_P_ey))
    
    gari_priors_keep_xor = np.concatenate([
        P_ez, P_ex, P_ey,
        P_ez_prime_xor, P_ex_prime_xor
    ])
    
    # --- Mode O: ez, ex, ey Scaled, ez', ex' XOR Aggregated ---
    gari_priors_scaled_xor = np.concatenate([
        prob(lam * C_z), prob(lam * C_x), prob(lam * C_y),
        P_ez_prime_xor, P_ex_prime_xor
    ])
    
    # --- LP Setup ---
    from scipy.sparse import bmat, eye as speye, vstack, hstack
    from scipy.optimize import linprog
    
    P_real_orig = np.concatenate([P_dx, P_dz, P_y])
    b_ub = cost(P_real_orig)
    
    N_real = len(b_ub)
    N_virtual = nx + nz
    
    I_nx = speye(nx, format='csc')
    I_nz = speye(nz, format='csc')
    
    # Original A_ub mapping virtuals to reals
    A_ub_orig = bmat([
        [I_nx, None],
        [None, I_nz],
        [U.T,  V.T]
    ], format='csc')
    
    # We add a variable 't' at the end of the variable vector: x = [g_virt, t]
    # Constraint 1: g_real >= t  =>  A_ub_orig * g_virt + t <= b_ub
    col_ones_real = np.ones((N_real, 1))
    A_upper = hstack([A_ub_orig, col_ones_real], format='csc')
    
    # Constraint 2: g_virt >= t  =>  -I * g_virt + t <= 0
    I_virt = speye(N_virtual, format='csc')
    col_ones_virt = np.ones((N_virtual, 1))
    A_lower = hstack([-I_virt, col_ones_virt], format='csc')
    
    # Combine all constraints
    A_ub_new = vstack([A_upper, A_lower], format='csc')
    b_ub_new = np.concatenate([b_ub, np.zeros(N_virtual)])
    bounds = [(0, None)] * (N_virtual + 1)
    lambda_reg = 1e-4

    # --- Mode Q: Linear Programming based weight distribution (No Lambda) ---
    c_obj_Q = np.zeros(N_virtual + 1)
    c_obj_Q[-1] = -1.0
    
    print("Solving LP (Max-Min Formulation)...")
    res_lp_Q = linprog(c_obj_Q, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_Q.success:
        x_res_Q = res_lp_Q.x
        g_virt_Q = x_res_Q[:-1]
        t_val_Q = x_res_Q[-1]
        
        print(f"LP Solver Finished. Minimum edge floor (t): {t_val_Q:.4f}")
        
        g_real_Q = b_ub - A_ub_orig.dot(g_virt_Q)
        
        P_real_Q = prob(g_real_Q)
        P_virt_Q = prob(g_virt_Q)
        
        gari_priors_mode_Q = np.concatenate([P_real_Q, P_virt_Q])
    else:
        print(f"LP Solver failed: {res_lp_Q.message}")
        raise NotImplementedError

    # --- Mode P: Fixed Epsilon LP ---
    c_obj_P = np.zeros(N_real + N_virtual)
    c_obj_P[-N_virtual:] = -1.0
    
    A_eq_P = hstack([speye(N_real, format='csc'), A_ub_orig], format='csc')
    b_eq_P = b_ub
    
    w_z, w_x, w_y = C_z, C_x, C_y
    min_w_vals = [np.min(w) for w in (w_z, w_x, w_y) if len(w) > 0]
    min_w = min(min_w_vals) if min_w_vals else 1e-10
    dynamic_eps = min_w / 1e5
    
    bounds_P = [(dynamic_eps, None)] * (N_real + N_virtual)
    
    print("Solving LP (Mode P: Fixed Epsilon)...")
    res_lp_P = linprog(c_obj_P, A_eq=A_eq_P, b_eq=b_eq_P, bounds=bounds_P, method='highs')
    
    if res_lp_P.success:
        x_res_P = res_lp_P.x
        g_real_P = x_res_P[:N_real]
        g_virt_P = x_res_P[N_real:]
        
        P_real_P = prob(g_real_P)
        P_virt_P = prob(g_virt_P)
        
        gari_priors_mode_P = np.concatenate([P_real_P, P_virt_P])
        print("LP Solver Mode P Finished.")
    else:
        print(f"Warning: LP Solver Mode P failed: {res_lp_P.message}")
        # Return original weights (aggregated) so decoder doesn't crash
        gari_priors_mode_P = gari_priors_agg.copy()


    # --- Mode R: Linear Programming based weight distribution (Uniform Lambda) ---
    c_obj_R = -lambda_reg * np.ones(N_virtual + 1)
    c_obj_R[-1] = -1.0
    
    print("Solving LP (Max-Min Formulation with Uniform Lambda)...")
    res_lp_R = linprog(c_obj_R, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_R.success:
        x_res_R = res_lp_R.x
        g_virt_R = x_res_R[:-1]
        t_val_R = x_res_R[-1]
        
        print(f"LP Solver Finished. Minimum edge floor (t): {t_val_R:.4f}")
        
        g_real_R = b_ub - A_ub_orig.dot(g_virt_R)
        
        P_real_R = prob(g_real_R)
        P_virt_R = prob(g_virt_R)
        
        gari_priors_mode_R = np.concatenate([P_real_R, P_virt_R])
    else:
        print(f"LP Solver failed: {res_lp_R.message}")
        raise NotImplementedError

    # --- Mode S: Linear Programming based weight distribution (Weighted Lambda) ---
    weights_virtual = b_ub[:N_virtual]
    lambda_array = lambda_reg * (weights_virtual / np.max(weights_virtual))
    c_obj_S = np.zeros(N_virtual + 1)
    c_obj_S[:-1] = -lambda_array
    c_obj_S[-1] = -1.0
    
    print("Solving LP (Max-Min Formulation with Weighted Lambda)...")
    res_lp_S = linprog(c_obj_S, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_S.success:
        x_res_S = res_lp_S.x
        g_virt_S = x_res_S[:-1]
        t_val_S = x_res_S[-1]
        
        print(f"LP Solver Finished. Minimum edge floor (t): {t_val_S:.4f}")
        
        g_real_S = b_ub - A_ub_orig.dot(g_virt_S)
        
        P_real_S = prob(g_real_S)
        P_virt_S = prob(g_virt_S)
        
        gari_priors_mode_S = np.concatenate([P_real_S, P_virt_S])
    else:
        print(f"LP Solver failed: {res_lp_S.message}")
        raise NotImplementedError


    # --- Mode S2: Max-Min Formulation (g_real >= t, Topologically Weighted Lambda) ---
    W_z = b_ub[:nx]
    W_x = b_ub[nx:N_virtual]
    W_y = b_ub[N_virtual:]
    
    S_z = W_z + U.dot(W_y)
    S_x = W_x + V.dot(W_y)
    S_virt = np.concatenate([S_z, S_x])
    lambda_array_S2 = lambda_reg * (S_virt / np.max(S_virt))
    
    c_obj_S2 = np.zeros(N_virtual + 1)
    c_obj_S2[:-1] = -lambda_array_S2
    c_obj_S2[-1] = -1.0
    
    print("Solving LP (Mode S2: Max-Min Formulation, Topologically Weighted Lambda)...")
    res_lp_S2 = linprog(c_obj_S2, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_S2.success:
        x_res_S2 = res_lp_S2.x
        g_virt_S2 = x_res_S2[:-1]
        g_real_S2 = b_ub - A_ub_orig.dot(g_virt_S2)
        gari_priors_mode_S2 = np.concatenate([prob(g_real_S2), prob(g_virt_S2)])
    else:
        print(f"LP Solver failed for Mode S2: {res_lp_S2.message}")
        raise NotImplementedError


    # --- Mode SO: Max-Min Formulation (g_real >= t, Original Weighted Lambda, Dynamic Safe Lambda Reg) ---
    weights_virtual = b_ub[:N_virtual]
    normalized_W = weights_virtual / np.max(weights_virtual)
    safe_lambda_reg_SO = 0.99 / np.sum(normalized_W)
    lambda_array_SO = safe_lambda_reg_SO * normalized_W
    
    c_obj_SO = np.zeros(N_virtual + 1)
    c_obj_SO[:-1] = -lambda_array_SO
    c_obj_SO[-1] = -1.0
    
    print("Solving LP (Mode SO: Max-Min Formulation, Original Weighted Lambda, Safe Lambda Reg)...")
    res_lp_SO = linprog(c_obj_SO, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_SO.success:
        x_res_SO = res_lp_SO.x
        g_virt_SO = x_res_SO[:-1]
        g_real_SO = b_ub - A_ub_orig.dot(g_virt_SO)
        gari_priors_mode_SO = np.concatenate([prob(g_real_SO), prob(g_virt_SO)])
    else:
        print(f"LP Solver failed for Mode SO: {res_lp_SO.message}")
        raise NotImplementedError

    # --- Mode SO2: Max-Min Formulation (g_real >= t, Topologically Weighted Lambda, Dynamic Safe Lambda Reg) ---
    W_z = b_ub[:nx]
    W_x = b_ub[nx:N_virtual]
    W_y = b_ub[N_virtual:]
    
    S_z_o = W_z + U.dot(W_y)
    S_x_o = W_x + V.dot(W_y)
    S_virt_o = np.concatenate([S_z_o, S_x_o])
    
    normalized_S = S_virt_o / np.max(S_virt_o)
    safe_lambda_reg_SO2 = 0.99 / np.sum(normalized_S)
    lambda_array_SO2 = safe_lambda_reg_SO2 * normalized_S
    
    c_obj_SO2 = np.zeros(N_virtual + 1)
    c_obj_SO2[:-1] = -lambda_array_SO2
    c_obj_SO2[-1] = -1.0
    
    print("Solving LP (Mode SO2: Max-Min Formulation, Topologically Weighted Lambda, Safe Lambda Reg)...")
    res_lp_SO2 = linprog(c_obj_SO2, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
    
    if res_lp_SO2.success:
        x_res_SO2 = res_lp_SO2.x
        g_virt_SO2 = x_res_SO2[:-1]
        g_real_SO2 = b_ub - A_ub_orig.dot(g_virt_SO2)
        gari_priors_mode_SO2 = np.concatenate([prob(g_real_SO2), prob(g_virt_SO2)])
    else:
        print(f"LP Solver failed for Mode SO2: {res_lp_SO2.message}")
        raise NotImplementedError

    # --- Mode U: Max-Min Formulation (Zero-Cost Reals allowed, Lambda = 1) ---
    # We remove the g_real >= t constraint. We only enforce g_virt >= t (A_lower).
    # Upper bound is just A_ub_orig * g_virt <= b_ub
    A_upper_UV = hstack([A_ub_orig, np.zeros((N_real, 1))], format='csc')
    A_ub_UV = vstack([A_upper_UV, A_lower], format='csc')
    
    # Enforce safety epsilon for A* (preventing exactly 0-cost edges)
    # 1. Virtual edges are >= t, so we bound t >= dynamic_eps
    bounds_UV = [(0, None)] * N_virtual + [(dynamic_eps, None)]
    # 2. Real edges are g_real = b_ub - A_ub_orig*g_virt >= eps => A_ub_orig*g_virt <= b_ub - eps
    b_ub_UV = np.concatenate([b_ub - dynamic_eps, np.zeros(N_virtual)])
    
    c_obj_U = np.zeros(N_virtual + 1)
    c_obj_U[:-1] = -1.0  # lambda = 1.0 for all virtual edges
    c_obj_U[-1] = -1.0
    
    print("Solving LP (Mode U: Zero-Cost Reals, Lambda = 1, Safety Eps)...")
    res_lp_U = linprog(c_obj_U, A_ub=A_ub_UV, b_ub=b_ub_UV, bounds=bounds_UV, method='highs')
    
    if res_lp_U.success:
        g_virt_U = res_lp_U.x[:-1]
        g_real_U = b_ub - A_ub_orig.dot(g_virt_U)
        gari_priors_mode_U = np.concatenate([prob(g_real_U), prob(g_virt_U)])
    else:
        print(f"LP Solver failed for Mode U: {res_lp_U.message}")
        raise NotImplementedError

    # --- Mode V: Max-Min Formulation (Zero-Cost Reals allowed, Topologically Weighted Lambda) ---
    W_z = b_ub[:nx]
    W_x = b_ub[nx:N_virtual]
    W_y = b_ub[N_virtual:]
    
    S_z = W_z + U.dot(W_y)
    S_x = W_x + V.dot(W_y)
    
    S_virt = np.concatenate([S_z, S_x])
    lambda_array_V = lambda_reg * (S_virt / np.max(S_virt))
    
    c_obj_V = np.zeros(N_virtual + 1)
    c_obj_V[:-1] = -lambda_array_V
    c_obj_V[-1] = -1.0
    
    print("Solving LP (Mode V: Zero-Cost Reals, Topologically Weighted Lambda, Safety Eps)...")
    res_lp_V = linprog(c_obj_V, A_ub=A_ub_UV, b_ub=b_ub_UV, bounds=bounds_UV, method='highs')
    
    if res_lp_V.success:
        g_virt_V = res_lp_V.x[:-1]
        g_real_V = b_ub - A_ub_orig.dot(g_virt_V)
        gari_priors_mode_V = np.concatenate([prob(g_real_V), prob(g_virt_V)])
    else:
        print(f"LP Solver failed for Mode V: {res_lp_V.message}")
        raise NotImplementedError
    
    x_orig_indices = np.where(det_types == 1)[0]
    z_orig_indices = np.where(det_types == 3)[0]

    time_vy = [max(
        np.max([x_orig_indices[r] for r in hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0,
        np.max([z_orig_indices[r] for r in hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0
    ) for c in i_hy]
    
    U_csr = U.tocsr()
    V_csr = V.tocsr()
    
    time_vx_old = [np.max([x_orig_indices[r] for r in hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0 for c in i_hx_only]
    time_vz_old = [np.max([z_orig_indices[r] for r in hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0 for c in i_hz_only]

    time_vx_new = [max(
        time_vx_old[i],
        max([time_vy[k] for k in U_csr.indices[U_csr.indptr[i]:U_csr.indptr[i+1]]]) if U_csr.indptr[i+1] > U_csr.indptr[i] else 0
    ) for i, c in enumerate(i_hx_only)]
    
    time_vz_new = [max(
        time_vz_old[i],
        max([time_vy[k] for k in V_csr.indices[V_csr.indptr[i]:V_csr.indptr[i+1]]]) if V_csr.indptr[i+1] > V_csr.indptr[i] else 0
    ) for i, c in enumerate(i_hz_only)]
    
    time_vx_min_old = [np.min([x_orig_indices[r] for r in hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0 for c in i_hx_only]
    time_vz_min_old = [np.min([z_orig_indices[r] for r in hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0 for c in i_hz_only]
    
    time_vy_min = []
    for c in i_hy:
        t_hx = np.min([x_orig_indices[r] for r in hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else float('inf')
        t_hz = np.min([z_orig_indices[r] for r in hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else float('inf')
        t_min = min(t_hx, t_hz)
        time_vy_min.append(t_min if t_min != float('inf') else 0)
        
    time_vx_min_new = []
    for i, c in enumerate(i_hx_only):
        t_y = [time_vy_min[k] for k in U_csr.indices[U_csr.indptr[i]:U_csr.indptr[i+1]]]
        min_y = min(t_y) if t_y else float('inf')
        t_min = min(time_vx_min_old[i], min_y)
        time_vx_min_new.append(t_min if t_min != float('inf') else 0)
        
    time_vz_min_new = []
    for i, c in enumerate(i_hz_only):
        t_y = [time_vy_min[k] for k in V_csr.indices[V_csr.indptr[i]:V_csr.indptr[i+1]]]
        min_y = min(t_y) if t_y else float('inf')
        t_min = min(time_vz_min_old[i], min_y)
        time_vz_min_new.append(t_min if t_min != float('inf') else 0)
    
    if return_dem:
        return matrices_to_dem(gari_matrix, gari_obs_matrix, gari_priors_agg), nx, nz, time_vx_old, time_vz_old, gari_obs_matrix_og, time_vx_new, time_vz_new, time_vx_min_old, time_vz_min_old, time_vx_min_new, time_vz_min_new
    return gari_matrix, gari_obs_matrix, gari_priors_agg, gari_priors_keep_free, gari_priors_keep_scaled, gari_priors_hidden_free, gari_priors_tiny, gari_priors_hf_agg, gari_priors_tiny_agg, gari_priors_keep_keep, gari_priors_keep_max, gari_priors_hf_max, gari_priors_tiny_max, gari_priors_keep_freeY_agg, gari_priors_keep_freeY_max, gari_priors_keep_xor, gari_priors_scaled_xor, gari_priors_mode_P, gari_priors_mode_Q, gari_priors_mode_R, gari_priors_mode_S, dx, dz, nx, nz, time_vx_old, time_vz_old, gari_obs_matrix_og, time_vx_new, time_vz_new, time_vx_min_old, time_vz_min_old, time_vx_min_new, time_vz_min_new, gari_priors_mode_U, gari_priors_mode_V, gari_priors_mode_S2, gari_priors_mode_SO, gari_priors_mode_SO2

def get_gari_orderings(dem, gari_dem, dx, dz, det_types, nx_virt, time_vx_old, time_vz_old, time_vx_new, time_vz_new, time_vx_min_old, time_vz_min_old, time_vx_min_new, time_vz_min_new):
    """
    Generates different detector processing orders for Tesseract's A* Search.
    """
    nx_real = np.sum(det_types == 1)
    nz_real = np.sum(det_types == 3)
    
    real_x_gari = list(range(0, nx_real))
    real_z_gari = list(range(nx_real, nx_real + nz_real))
    virt_x_gari = list(range(nx_real + nz_real, nx_real + nz_real + nx_virt))
    virt_z_gari = list(range(nx_real + nz_real + nx_virt, gari_dem.num_detectors))
    
    x_orig_indices = np.where(det_types == 1)[0]
    z_orig_indices = np.where(det_types == 3)[0]

    x_map = {orig_idx: int(gari_idx) for gari_idx, orig_idx in enumerate(x_orig_indices)}
    z_map = {orig_idx: int(nx_real + gari_idx) for gari_idx, orig_idx in enumerate(z_orig_indices)}

    # Standard Block Combinations
    order_1 = real_x_gari + virt_x_gari + real_z_gari + virt_z_gari
    order_2 = real_x_gari + real_z_gari + virt_x_gari + virt_z_gari
    order_3 = real_x_gari + virt_z_gari + real_z_gari + virt_x_gari
    order_4 = real_x_gari + real_z_gari + virt_z_gari + virt_x_gari
    order_8 = real_x_gari + virt_x_gari + virt_z_gari + real_z_gari
    order_9 = real_z_gari + real_x_gari + virt_z_gari + virt_x_gari
    order_10 = real_z_gari + real_x_gari + virt_x_gari + virt_z_gari

    # Option 7: Chronological Real, Chronological Virtual
    real_gari_chronological = []
    for orig_idx in range(dem.num_detectors):
        if orig_idx in x_map:
            real_gari_chronological.append(x_map[orig_idx])
        elif orig_idx in z_map:
            real_gari_chronological.append(z_map[orig_idx])
            
    virt_with_time_old = []
    for c in range(nx_virt):
        virt_with_time_old.append((time_vx_old[c], nx_real + nz_real + c))
    for c in range(len(time_vz_old)):
        virt_with_time_old.append((time_vz_old[c], nx_real + nz_real + nx_virt + c))
        
    virt_with_time_old.sort(key=lambda x: x[0])
    virt_gari_chronological_old = [v[1] for v in virt_with_time_old]
    
    order_7 = real_gari_chronological + virt_gari_chronological_old

    # Option 7a: Chronological Real, Chronological Virtual (with e_y time incorporated)
    virt_with_time_new = []
    for c in range(nx_virt):
        virt_with_time_new.append((time_vx_new[c], nx_real + nz_real + c))
    for c in range(len(time_vz_new)):
        virt_with_time_new.append((time_vz_new[c], nx_real + nz_real + nx_virt + c))
        
    virt_with_time_new.sort(key=lambda x: x[0])
    virt_gari_chronological_new = [v[1] for v in virt_with_time_new]
    
    order_7a = real_gari_chronological + virt_gari_chronological_new
    
    # Option 7b: Chronological Real, Chronological Virtual (min_t, no e_y)
    virt_with_time_min_old = []
    for c in range(nx_virt):
        virt_with_time_min_old.append((time_vx_min_old[c], nx_real + nz_real + c))
    for c in range(len(time_vz_min_old)):
        virt_with_time_min_old.append((time_vz_min_old[c], nx_real + nz_real + nx_virt + c))
        
    virt_with_time_min_old.sort(key=lambda x: x[0])
    order_7b = real_gari_chronological + [v[1] for v in virt_with_time_min_old]
    
    # Option 7c: Chronological Real, Chronological Virtual (min_t, with e_y)
    virt_with_time_min_new = []
    for c in range(nx_virt):
        virt_with_time_min_new.append((time_vx_min_new[c], nx_real + nz_real + c))
    for c in range(len(time_vz_min_new)):
        virt_with_time_min_new.append((time_vz_min_new[c], nx_real + nz_real + nx_virt + c))
        
    virt_with_time_min_new.sort(key=lambda x: x[0])
    order_7c = real_gari_chronological + [v[1] for v in virt_with_time_min_new]

    # Helper for Order 11 (Safe-Interleaved) Variants
    def get_order_11_variant(v_x_times, v_z_times):
        all_det = []
        for orig_idx in range(dem.num_detectors):
            if orig_idx in x_map:
                all_det.append((orig_idx, x_map[orig_idx]))
            elif orig_idx in z_map:
                all_det.append((orig_idx, z_map[orig_idx]))
        
        for c in range(nx_virt):
            all_det.append((v_x_times[c] + 0.1, nx_real + nz_real + c))
        for c in range(len(v_z_times)):
            all_det.append((v_z_times[c] + 0.1, nx_real + nz_real + nx_virt + c))
            
        all_det.sort(key=lambda x: x[0])
        return [v[1] for v in all_det]

    order_11 = get_order_11_variant(time_vx_new, time_vz_new)       # Safe-Interleaved (includes Y) - Best
    order_11a = get_order_11_variant(time_vx_old, time_vz_old)      # Last-Touch Interleaved (ignores Y)
    order_11b = get_order_11_variant(time_vx_min_new, time_vz_min_new) # First-Touch Interleaved (includes Y)

    # Helper for Order 12 (Reverse Safe-Interleaved) Variants
    def get_order_12_variant(v_x_times, v_z_times):
        all_det = []
        for orig_idx in range(dem.num_detectors):
            if orig_idx in x_map:
                all_det.append((orig_idx, x_map[orig_idx]))
            elif orig_idx in z_map:
                all_det.append((orig_idx, z_map[orig_idx]))
        
        for c in range(nx_virt):
            all_det.append((v_x_times[c] - 0.1, nx_real + nz_real + c))
        for c in range(len(v_z_times)):
            all_det.append((v_z_times[c] - 0.1, nx_real + nz_real + nx_virt + c))
            
        all_det.sort(key=lambda x: x[0], reverse=True)
        return [v[1] for v in all_det]

    order_12 = get_order_12_variant(time_vx_min_new, time_vz_min_new)       # Safe-Reverse (includes Y) - Best for reverse
    order_12a = get_order_12_variant(time_vx_min_old, time_vz_min_old)      # Last-Touch Reverse (ignores Y)
    order_12b = get_order_12_variant(time_vx_new, time_vz_new)              # First-Touch Reverse (includes Y)

    return {
        # "order1": order_1, # Option 1 (RealX, VirtX, RealZ, VirtZ)
        "order2": order_2, # Option 2 (RealX, RealZ, VirtX, VirtZ)
        "order4": order_4, # Option 4 (RealX, RealZ, VirtZ, VirtX)
        "order7": order_7, # Option 7 (Chrono Real, Chrono Virt without ey)
        # "order7a": order_7a, # Option 7a (Chrono Real, Chrono Virt with ey max)
        # "order7b": order_7b, # Option 7b (Chrono Real, Chrono Virt min without ey)
        # "order7c": order_7c, # Option 7c (Chrono Real, Chrono Virt min with ey)
        "order9": order_9, # Option 9 (RealZ, RealX, VirtZ, VirtX)
        "order10": order_10, # Option 10 (RealZ, RealX, VirtX, VirtZ)
        # "order11": order_11, # Option 11 (Safe-Interleaved Chrono with ey) - BEST FORWARD
        # "order11a": order_11a, # Option 11a (Last-Touch Interleaved Chrono without ey)
        # "order11b": order_11b, # Option 11b (First-Touch Interleaved Chrono with ey)
        # "order12": order_12, # Option 12 (Safe-Reverse Interleaved with ey) - BEST REVERSE
        # "order12a": order_12a, # Option 12a (Last-Touch Reverse Interleaved without ey)
        # "order12b": order_12b # Option 12b (First-Touch Reverse Interleaved with ey)
    }


def process_directory(input_path):
    """
    Recursively find all *.stim files matching input_path.
    For each file:
    - Load the circuit
    - Generate DEM
    - Apply gari_transform(return_dem=False)
    - Save DEMs for Mode A, Mode F, Mode J to disk next to the .stim file
    - Compute mapping.json and save it
    """
    if os.path.isdir(input_path):
        stim_files = glob.glob(os.path.join(input_path, "**", "*.stim"), recursive=True)
    else:
        if "*" not in input_path:
            input_path += "*"
        stim_files = [f for f in glob.glob(input_path, recursive=True) if f.endswith(".stim")]
    for stim_path in stim_files:
        print(f"Processing {stim_path}")
        try:
            # Filter logic: if path contains color code, only process 'superdense_color_code'
            if "color_code" in stim_path or "colorcodes" in stim_path:
                if "superdense_color_code" not in os.path.basename(stim_path):
                    continue

            circuit = stim.Circuit.from_file(stim_path)
            dem = circuit.detector_error_model(
                decompose_errors=False, 
                flatten_loops=True, 
                ignore_decomposition_failures=True
            )
            det_types = get_detector_types(circuit)
            H, L, priors, errors = dem_to_check_matrices(dem)
            res = gari_transform(H, L, det_types, priors, return_dem=False)
            
            gari_matrix = res[0]
            gari_obs_matrix = res[1]
            gari_obs_matrix_og = res[27]
            
            stim_dir = os.path.dirname(stim_path)
            stim_stem = os.path.splitext(os.path.basename(stim_path))[0]
            gari_dir = os.path.join(stim_dir, "gari")
            os.makedirs(gari_dir, exist_ok=True)
            base_path = os.path.join(gari_dir, stim_stem)
            
            modes_to_generate = {
                "modeA": 2,
                "modeB": 3,
                "modeC": 4,
                "modeF": 7,
                "modeG": 8,
                "modeH": 9,
                "modeI": 10,
                "modeJ": 11,
                "modeK": 12,
                "modeL": 13,
                "modeM": 14,
                "modeN": 15,
                "modeO": 16,
                "modeP": 17,
                "modeQ": 18,
                "modeR": 19,
                "modeS": 20,
                "modeU": 34,
                "modeV": 35,
                "modeS2": 36,
                "modeSO": 37,
                "modeSO2": 38
            }
            
            for mode_name, index in modes_to_generate.items():
                priors_for_mode = res[index]
                dem_for_mode = matrices_to_dem(gari_matrix, gari_obs_matrix, priors_for_mode)
                dem_for_mode.to_file(base_path + f"_{mode_name}.dem")
                
            for mode_name, index in modes_to_generate.items():
                priors_for_mode = res[index]
                dem_for_mode_og = matrices_to_dem(gari_matrix, gari_obs_matrix_og, priors_for_mode)
                dem_for_mode_og.to_file(base_path + f"_ogL_{mode_name}.dem")
            
            x_orig_indices = np.where(det_types == 1)[0]
            z_orig_indices = np.where(det_types == 3)[0]
            
            nx = len(x_orig_indices)
            
            mapping = [-1] * dem.num_detectors
            for gari_idx, orig_idx in enumerate(x_orig_indices):
                mapping[int(orig_idx)] = int(gari_idx)
            for gari_idx, orig_idx in enumerate(z_orig_indices):
                mapping[int(orig_idx)] = int(nx + gari_idx)
                
            dummy_dem = matrices_to_dem(res[0], res[1], res[2])
            orderings = get_gari_orderings(dem, dummy_dem, res[21], res[22], det_types, res[23], res[25], res[26], res[28], res[29], res[30], res[31], res[32], res[33])
            
            det_orders_clean = {}
            for k, v in orderings.items():
                det_orders_clean[k] = [int(x) for x in v]
                
            mapping_dict = {
                "num_original_detectors": dem.num_detectors,
                "mapping": mapping,
                "det_orders": det_orders_clean
            }
            with open(base_path + "_mapping.json", "w") as f:
                json.dump(mapping_dict, f, indent=2)
                
            print(f"Successfully processed {stim_path}")
        except Exception as e:
            print(f"Error processing {stim_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process stim files and generate Gari DEMs.")
    parser.add_argument("input_path", type=str, help="Input directory, file prefix, or glob pattern for .stim files")
    args = parser.parse_args()
    
    process_directory(args.input_path)
