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
    
    time_vy = [max(
        np.max(hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0,
        np.max(hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0
    ) for c in i_hy]
    
    U_csr = U.tocsr()
    V_csr = V.tocsr()
    
    time_vx_old = [np.max(hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0 for c in i_hx_only]
    time_vz_old = [np.max(hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0 for c in i_hz_only]

    time_vx_new = [max(
        time_vx_old[i],
        max([time_vy[k] for k in U_csr.indices[U_csr.indptr[i]:U_csr.indptr[i+1]]]) if U_csr.indptr[i+1] > U_csr.indptr[i] else 0
    ) for i, c in enumerate(i_hx_only)]
    
    time_vz_new = [max(
        time_vz_old[i],
        max([time_vy[k] for k in V_csr.indices[V_csr.indptr[i]:V_csr.indptr[i+1]]]) if V_csr.indptr[i+1] > V_csr.indptr[i] else 0
    ) for i, c in enumerate(i_hz_only)]
    
    time_vx_min_old = [np.min(hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0 for c in i_hx_only]
    time_vz_min_old = [np.min(hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0 for c in i_hz_only]
    
    time_vy_min = []
    for c in i_hy:
        t_hx = np.min(hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else float('inf')
        t_hz = np.min(hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else float('inf')
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
    return gari_matrix, gari_obs_matrix, gari_priors_agg, gari_priors_keep_free, gari_priors_keep_scaled, gari_priors_hidden_free, gari_priors_tiny, gari_priors_hf_agg, gari_priors_tiny_agg, gari_priors_keep_keep, gari_priors_keep_max, gari_priors_hf_max, gari_priors_tiny_max, gari_priors_keep_freeY_agg, gari_priors_keep_freeY_max, gari_priors_keep_xor, dx, dz, nx, nz, time_vx_old, time_vz_old, gari_obs_matrix_og, time_vx_new, time_vz_new, time_vx_min_old, time_vz_min_old, time_vx_min_new, time_vz_min_new

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

    # Option 5: First-Touch Interleaved (Chronological)
    virtuals_to_insert_first = {i: [] for i in range(dem.num_detectors)}
    for c in range(dx.shape[1]):
        start = dx.indptr[c]
        end = dx.indptr[c+1]
        if start < end:
            min_orig_idx = min([x_orig_indices[r] for r in dx.indices[start:end]])
            virtuals_to_insert_first[min_orig_idx].append(nx_real + nz_real + c)
        else:
            virtuals_to_insert_first[0].append(nx_real + nz_real + c)
    for c in range(dz.shape[1]):
        start = dz.indptr[c]
        end = dz.indptr[c+1]
        if start < end:
            min_orig_idx = min([z_orig_indices[r] for r in dz.indices[start:end]])
            virtuals_to_insert_first[min_orig_idx].append(nx_real + nz_real + nx_virt + c)
        else:
            virtuals_to_insert_first[0].append(nx_real + nz_real + nx_virt + c)
            
    order_5 = []
    for orig_idx in range(dem.num_detectors):
        if orig_idx in x_orig_indices:
            gari_idx = np.where(x_orig_indices == orig_idx)[0][0]
            order_5.append(int(gari_idx))
        elif orig_idx in z_orig_indices:
            gari_idx = np.where(z_orig_indices == orig_idx)[0][0]
            order_5.append(int(nx_real + gari_idx))
        order_5.extend(virtuals_to_insert_first[orig_idx])

    # Option 6: Last-Touch Interleaved (Chronological)
    virtuals_to_insert_last = {i: [] for i in range(dem.num_detectors)}
    for c in range(dx.shape[1]):
        start = dx.indptr[c]
        end = dx.indptr[c+1]
        if start < end:
            max_orig_idx = max([x_orig_indices[r] for r in dx.indices[start:end]])
            virtuals_to_insert_last[max_orig_idx].append(nx_real + nz_real + c)
        else:
            virtuals_to_insert_last[dem.num_detectors-1].append(nx_real + nz_real + c)
    for c in range(dz.shape[1]):
        start = dz.indptr[c]
        end = dz.indptr[c+1]
        if start < end:
            max_orig_idx = max([z_orig_indices[r] for r in dz.indices[start:end]])
            virtuals_to_insert_last[max_orig_idx].append(nx_real + nz_real + nx_virt + c)
        else:
            virtuals_to_insert_last[dem.num_detectors-1].append(nx_real + nz_real + nx_virt + c)
            
    order_6 = []
    for orig_idx in range(dem.num_detectors):
        if orig_idx in x_orig_indices:
            gari_idx = np.where(x_orig_indices == orig_idx)[0][0]
            order_6.append(int(gari_idx))
        elif orig_idx in z_orig_indices:
            gari_idx = np.where(z_orig_indices == orig_idx)[0][0]
            order_6.append(int(nx_real + gari_idx))
        order_6.extend(virtuals_to_insert_last[orig_idx])

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
        if orig_idx in x_orig_indices:
            gari_idx = np.where(x_orig_indices == orig_idx)[0][0]
            real_gari_chronological.append(int(gari_idx))
        elif orig_idx in z_orig_indices:
            gari_idx = np.where(z_orig_indices == orig_idx)[0][0]
            real_gari_chronological.append(int(nx_real + gari_idx))
            
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

    return {
        # "order1": order_1, # Option 1 (RealX, VirtX, RealZ, VirtZ)
        "order2": order_2, # Option 2 (RealX, RealZ, VirtX, VirtZ)
        "order4": order_4, # Option 4 (RealX, RealZ, VirtZ, VirtX)
        #"order5": order_5, # Option 5 (First-Touch Interleaved Chrono) - EXTREMELY SLOW
        #"order6": order_6, # Option 6 (Last-Touch Interleaved Chrono) - EXTREMELY SLOW
        "order7": order_7, # Option 7 (Chrono Real, Chrono Virt without ey)
        "order7a": order_7a, # Option 7a (Chrono Real, Chrono Virt with ey max)
        "order7b": order_7b, # Option 7b (Chrono Real, Chrono Virt min without ey)
        "order7c": order_7c, # Option 7c (Chrono Real, Chrono Virt min with ey)
        "order9": order_9, # Option 9 (RealZ, RealX, VirtZ, VirtX)
        "order10": order_10 # Option 10 (RealZ, RealX, VirtX, VirtZ)
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
            gari_obs_matrix_og = res[22]
            
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
                "modeN": 15
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
            orderings = get_gari_orderings(dem, dummy_dem, res[16], res[17], det_types, res[18], res[20], res[21], res[23], res[24], res[25], res[26], res[27], res[28])
            
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
