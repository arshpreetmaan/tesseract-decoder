import os
import glob
import json
import argparse
import pathlib
import stim
import numpy as np
from scipy.sparse import csc_matrix

def get_target_path(relative_path):
    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    return os.path.join(workspace_root, relative_path)

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



def gari_transform(H: csc_matrix, L: csc_matrix, det_types: np.ndarray) -> dict:
    """
    Applies the Gari Transform (arXiv:2510.14060) to the check matrix.
    Splits Y errors into independent X and Z components and adds virtual detectors.
    """
    is_x_det = (det_types == 1)
    is_z_det = (det_types == 3)
    
    x_orig_indices = np.where(is_x_det)[0]
    z_orig_indices = np.where(is_z_det)[0]
    
    H_csr = H.tocsr()
    hx = H_csr[is_x_det, :]
    hz = H_csr[is_z_det, :]
    
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
    
    mx, nx = dx.shape
    mz, nz = dz.shape
    ny = hx_yonly.shape[1]
    
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
    
    blocks = [
        [None, None, None, dx,   None],
        [None, None, None, None, dz  ],
        [I_nx, None, U,    I_nx, None],
        [None, I_nz, V,    None, I_nz]
    ]
    
    gari_matrix = bmat(blocks, format='csc', dtype=np.uint8)
    
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
    
    gari_structure = {
        "gari_matrix": gari_matrix,
        "gari_obs_matrix": gari_obs_matrix,
        "gari_obs_matrix_og": gari_obs_matrix_og,
        "U": U,
        "V": V,
        "nx_virt": nx,
        "nz_virt": nz,
        "nx_real": mx,
        "nz_real": mz,
        "num_original_detectors": H.shape[0],
        "num_gari_detectors": gari_matrix.shape[0],
        "i_hx_only": i_hx_only,
        "i_hz_only": i_hz_only,
        "i_hy": i_hy,
        "dx": dx,
        "dz": dz,
        "hx_csc": hx_csc,
        "hz_csc": hz_csc,
        "x_orig_indices": x_orig_indices,
        "z_orig_indices": z_orig_indices,
    }
    return gari_structure


def assign_prior_weights(gari_structure: dict, method: str, original_priors: np.ndarray) -> np.ndarray:
    i_hx_only = gari_structure["i_hx_only"]
    i_hz_only = gari_structure["i_hz_only"]
    i_hy = gari_structure["i_hy"]
    
    P_dx = original_priors[i_hx_only]
    P_dz = original_priors[i_hz_only]
    P_y = original_priors[i_hy]
    
    P_ez = P_dx
    P_ex = P_dz
    P_ey = P_y
    
    U = gari_structure["U"]
    V = gari_structure["V"]
    nx = gari_structure["nx_virt"]
    nz = gari_structure["nz_virt"]
    
    def cost(p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.log((1-p)/p)
        
    def prob(c):
        c = np.clip(c, -30, 30)
        return 1 / (1 + np.exp(c))
        
    P_ez_prime_agg = P_ez + U @ P_ey
    P_ex_prime_agg = P_ex + V @ P_ey
    
    if method == "modeA":
        return np.concatenate([P_ez, P_ex, P_ey, P_ez_prime_agg, P_ex_prime_agg])
        
    if method == "modeB":
        return np.concatenate([
            P_ez, P_ex, P_ey,
            np.full_like(P_ez, 0.499),
            np.full_like(P_ex, 0.499)
        ])
        
    lam = 0.001
    C_z_prime_agg = cost(P_ez_prime_agg)
    C_x_prime_agg = cost(P_ex_prime_agg)
    
    if method == "modeC":
        return np.concatenate([
            P_ez, P_ex, P_ey,
            prob(lam * C_z_prime_agg),
            prob(lam * C_x_prime_agg)
        ])
        
    if method == "modeD":
        return np.concatenate([
            np.full_like(P_ez, 0.499), 
            np.full_like(P_ex, 0.499), 
            np.full_like(P_ey, 0.499), 
            P_ez, P_ex
        ])
        
    C_z = cost(P_ez)
    C_x = cost(P_ex)
    C_y = cost(P_ey)
    
    if method == "modeE":
        return np.concatenate([
            prob(lam * C_z), 
            prob(lam * C_x), 
            prob(lam * C_y), 
            P_ez, P_ex
        ])
        
    if method == "modeF":
        return np.concatenate([
            np.full_like(P_ez, 0.499), 
            np.full_like(P_ex, 0.499), 
            np.full_like(P_ey, 0.499), 
            P_ez_prime_agg, P_ex_prime_agg
        ])
        
    if method == "modeG":
        return np.concatenate([
            prob(lam * C_z), 
            prob(lam * C_x), 
            prob(lam * C_y), 
            P_ez_prime_agg, P_ex_prime_agg
        ])
        
    if method == "modeH":
        return np.concatenate([
            P_ez, P_ex, P_ey, 
            P_ez, P_ex
        ])
        
    U_weighted = U.multiply(P_ey)
    max_ey_to_z = U_weighted.max(axis=1).toarray().flatten()
    P_ez_prime_max = np.maximum(P_ez, max_ey_to_z)

    V_weighted = V.multiply(P_ey)
    max_ey_to_x = V_weighted.max(axis=1).toarray().flatten()
    P_ex_prime_max = np.maximum(P_ex, max_ey_to_x)
    
    if method == "modeI":
        return np.concatenate([
            P_ez, P_ex, P_ey, 
            P_ez_prime_max, P_ex_prime_max
        ])
        
    if method == "modeJ":
        return np.concatenate([
            np.full_like(P_ez, 0.499), 
            np.full_like(P_ex, 0.499), 
            np.full_like(P_ey, 0.499), 
            P_ez_prime_max, P_ex_prime_max
        ])
        
    if method == "modeK":
        return np.concatenate([
            prob(lam * C_z), 
            prob(lam * C_x), 
            prob(lam * C_y), 
            P_ez_prime_max, P_ex_prime_max
        ])
        
    if method == "modeL":
        return np.concatenate([
            P_ez, P_ex, np.full_like(P_ey, 0.499),
            P_ez_prime_agg, P_ex_prime_agg
        ])
        
    if method == "modeM":
        return np.concatenate([
            P_ez, P_ex, np.full_like(P_ey, 0.499),
            P_ez_prime_max, P_ex_prime_max
        ])
        
    eps = 1e-15
    log_1m2_P_ez = np.log(np.clip(1 - 2 * P_ez, eps, 1.0))
    log_1m2_P_ex = np.log(np.clip(1 - 2 * P_ex, eps, 1.0))
    log_1m2_P_ey = np.log(np.clip(1 - 2 * P_ey, eps, 1.0))
    
    P_ez_prime_xor = 0.5 * (1 - np.exp(log_1m2_P_ez + U @ log_1m2_P_ey))
    P_ex_prime_xor = 0.5 * (1 - np.exp(log_1m2_P_ex + V @ log_1m2_P_ey))
    
    if method == "modeN":
        return np.concatenate([
            P_ez, P_ex, P_ey,
            P_ez_prime_xor, P_ex_prime_xor
        ])
        
    if method == "modeO":
        return np.concatenate([
            prob(lam * C_z), prob(lam * C_x), prob(lam * C_y),
            P_ez_prime_xor, P_ex_prime_xor
        ])

    from scipy.sparse import bmat, eye as speye, vstack, hstack
    from scipy.optimize import linprog
    
    P_real_orig = np.concatenate([P_dx, P_dz, P_y])
    b_ub = cost(P_real_orig)
    
    N_real = len(b_ub)
    N_virtual = nx + nz
    
    I_nx = speye(nx, format='csc')
    I_nz = speye(nz, format='csc')
    
    A_ub_orig = bmat([
        [I_nx, None],
        [None, I_nz],
        [U.T,  V.T]
    ], format='csc')

    if method == "modePR":
        if not np.all(np.isfinite(P_real_orig)) or not np.all(
            (P_real_orig > 0) & (P_real_orig <= 0.5)
        ):
            raise ValueError("modePR probabilities must be finite and in (0, 0.5]")
        source_costs = np.log1p(-P_real_orig) - np.log(P_real_orig)
        if N_virtual == 0:
            return P_real_orig.copy()

        # PR-273's lexicographic LP first maximizes a common floor across the
        # residual physical costs c - A g and the barred costs g.
        floor_constraints = bmat([
            [A_ub_orig, np.ones((N_real, 1))],
            [-speye(N_virtual, format='csc'), np.ones((N_virtual, 1))],
        ], format='csc')
        floor_objective = np.zeros(N_virtual + 1, dtype=np.float64)
        floor_objective[-1] = -1
        floor_result = linprog(
            floor_objective,
            A_ub=floor_constraints,
            b_ub=np.concatenate([source_costs, np.zeros(N_virtual)]),
            bounds=(0, None),
            method='highs',
        )
        if not floor_result.success:
            raise RuntimeError(
                "modePR floor solver failed: " + str(floor_result.message)
            )
        tolerance = 1e-7 * max(
            1.0, float(np.max(source_costs, initial=0.0))
        )
        cost_floor = max(0.0, float(floor_result.x[-1]) - tolerance)

        # Preserve that floor, then maximize the total barred cost.
        result = linprog(
            -np.ones(N_virtual, dtype=np.float64),
            A_ub=A_ub_orig,
            b_ub=source_costs - cost_floor,
            bounds=(cost_floor, None),
            method='highs',
        )
        if not result.success:
            raise RuntimeError(
                "modePR barred-cost solver failed: " + str(result.message)
            )
        barred_costs = np.asarray(result.x)
        residual_costs = source_costs - np.asarray(
            A_ub_orig @ barred_costs
        ).reshape(-1)
        if (
            np.min(barred_costs) < cost_floor - tolerance
            or np.min(residual_costs) < cost_floor - tolerance
        ):
            raise RuntimeError("modePR solver returned an infeasible solution")
        gari_costs = np.concatenate([
            np.maximum(residual_costs, 0.0),
            np.maximum(barred_costs, 0.0),
        ])
        return np.exp(-np.logaddexp(0, gari_costs))
    
    col_ones_real = np.ones((N_real, 1))
    A_upper = hstack([A_ub_orig, col_ones_real], format='csc')
    
    I_virt = speye(N_virtual, format='csc')
    col_ones_virt = np.ones((N_virtual, 1))
    A_lower = hstack([-I_virt, col_ones_virt], format='csc')
    
    A_ub_new = vstack([A_upper, A_lower], format='csc')
    b_ub_new = np.concatenate([b_ub, np.zeros(N_virtual)])
    bounds = [(0, None)] * (N_virtual + 1)
    lambda_reg = 1e-4

    if method == "modeQ":
        c_obj_Q = np.zeros(N_virtual + 1)
        c_obj_Q[-1] = -1.0
        res_lp_Q = linprog(c_obj_Q, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_Q.success:
            g_virt_Q = res_lp_Q.x[:-1]
            g_real_Q = b_ub - A_ub_orig.dot(g_virt_Q)
            return np.concatenate([prob(g_real_Q), prob(g_virt_Q)])
        raise NotImplementedError("LP modeQ failed")

    if method == "modeP":
        c_obj_P = np.zeros(N_real + N_virtual)
        c_obj_P[-N_virtual:] = -1.0
        A_eq_P = hstack([speye(N_real, format='csc'), A_ub_orig], format='csc')
        b_eq_P = b_ub
        w_z, w_x, w_y = C_z, C_x, C_y
        min_w_vals = [np.min(w) for w in (w_z, w_x, w_y) if len(w) > 0]
        min_w = min(min_w_vals) if min_w_vals else 1e-10
        dynamic_eps = min_w / 1e5
        bounds_P = [(dynamic_eps, None)] * (N_real + N_virtual)
        res_lp_P = linprog(c_obj_P, A_eq=A_eq_P, b_eq=b_eq_P, bounds=bounds_P, method='highs')
        if res_lp_P.success:
            return np.concatenate([prob(res_lp_P.x[:N_real]), prob(res_lp_P.x[N_real:])])
        return np.concatenate([P_ez, P_ex, P_ey, P_ez_prime_agg, P_ex_prime_agg])

    if method == "modeR":
        c_obj_R = -lambda_reg * np.ones(N_virtual + 1)
        c_obj_R[-1] = -1.0
        res_lp_R = linprog(c_obj_R, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_R.success:
            g_virt_R = res_lp_R.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_R)), prob(g_virt_R)])
        raise NotImplementedError("LP modeR failed")

    if method == "modeS":
        weights_virtual = b_ub[:N_virtual]
        lambda_array = lambda_reg * (weights_virtual / np.max(weights_virtual))
        c_obj_S = np.zeros(N_virtual + 1)
        c_obj_S[:-1] = -lambda_array
        c_obj_S[-1] = -1.0
        res_lp_S = linprog(c_obj_S, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_S.success:
            g_virt_S = res_lp_S.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_S)), prob(g_virt_S)])
        raise NotImplementedError("LP modeS failed")

    if method == "modeS2":
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
        res_lp_S2 = linprog(c_obj_S2, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_S2.success:
            g_virt_S2 = res_lp_S2.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_S2)), prob(g_virt_S2)])
        raise NotImplementedError("LP modeS2 failed")

    if method == "modeSO":
        weights_virtual = b_ub[:N_virtual]
        normalized_W = weights_virtual / np.max(weights_virtual)
        safe_lambda_reg_SO = 0.99 / np.sum(normalized_W)
        lambda_array_SO = safe_lambda_reg_SO * normalized_W
        c_obj_SO = np.zeros(N_virtual + 1)
        c_obj_SO[:-1] = -lambda_array_SO
        c_obj_SO[-1] = -1.0
        res_lp_SO = linprog(c_obj_SO, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_SO.success:
            g_virt_SO = res_lp_SO.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_SO)), prob(g_virt_SO)])
        raise NotImplementedError("LP modeSO failed")

    if method == "modeSO2":
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
        res_lp_SO2 = linprog(c_obj_SO2, A_ub=A_ub_new, b_ub=b_ub_new, bounds=bounds, method='highs')
        if res_lp_SO2.success:
            g_virt_SO2 = res_lp_SO2.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_SO2)), prob(g_virt_SO2)])
        raise NotImplementedError("LP modeSO2 failed")
        
    # Modes U and V
    w_z, w_x, w_y = C_z, C_x, C_y
    min_w_vals = [np.min(w) for w in (w_z, w_x, w_y) if len(w) > 0]
    min_w = min(min_w_vals) if min_w_vals else 1e-10
    dynamic_eps = min_w / 1e5

    A_upper_UV = hstack([A_ub_orig, np.zeros((N_real, 1))], format='csc')
    A_ub_UV = vstack([A_upper_UV, A_lower], format='csc')
    bounds_UV = [(0, None)] * N_virtual + [(dynamic_eps, None)]
    b_ub_UV = np.concatenate([b_ub - dynamic_eps, np.zeros(N_virtual)])

    if method == "modeU":
        c_obj_U = np.zeros(N_virtual + 1)
        c_obj_U[:-1] = -1.0
        c_obj_U[-1] = -1.0
        res_lp_U = linprog(c_obj_U, A_ub=A_ub_UV, b_ub=b_ub_UV, bounds=bounds_UV, method='highs')
        if res_lp_U.success:
            g_virt_U = res_lp_U.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_U)), prob(g_virt_U)])
        raise NotImplementedError("LP modeU failed")

    if method == "modeV":
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
        res_lp_V = linprog(c_obj_V, A_ub=A_ub_UV, b_ub=b_ub_UV, bounds=bounds_UV, method='highs')
        if res_lp_V.success:
            g_virt_V = res_lp_V.x[:-1]
            return np.concatenate([prob(b_ub - A_ub_orig.dot(g_virt_V)), prob(g_virt_V)])
        raise NotImplementedError("LP modeV failed")

    raise ValueError(f"Unknown method {method}")


def get_detector_orderings(gari_structure: dict, det_types: np.ndarray, ordering_name: str) -> list:
    hx_csc = gari_structure["hx_csc"]
    hz_csc = gari_structure["hz_csc"]
    U_csr = gari_structure["U"].tocsr()
    V_csr = gari_structure["V"].tocsr()
    
    i_hx_only = gari_structure["i_hx_only"]
    i_hz_only = gari_structure["i_hz_only"]
    i_hy = gari_structure["i_hy"]
    
    x_orig_indices = gari_structure["x_orig_indices"]
    z_orig_indices = gari_structure["z_orig_indices"]
    
    nx_virt = gari_structure["nx_virt"]
    nz_virt = gari_structure["nz_virt"]
    nx_real = gari_structure["nx_real"]
    nz_real = gari_structure["nz_real"]
    num_original_detectors = gari_structure["num_original_detectors"]
    num_gari_detectors = gari_structure["num_gari_detectors"]
    
    # Precompute time_vx_old, time_vz_old, time_vx_new, time_vz_new, time_vx_min_old, time_vz_min_old, time_vx_min_new, time_vz_min_new
    
    time_vy = [max(
        np.max([x_orig_indices[r] for r in hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0,
        np.max([z_orig_indices[r] for r in hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0
    ) for c in i_hy]
    
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
    
    # Common mappings
    real_x_gari = list(range(0, nx_real))
    real_z_gari = list(range(nx_real, nx_real + nz_real))
    virt_x_gari = list(range(nx_real + nz_real, nx_real + nz_real + nx_virt))
    virt_z_gari = list(range(nx_real + nz_real + nx_virt, num_gari_detectors))
    
    x_map = {orig_idx: int(gari_idx) for gari_idx, orig_idx in enumerate(x_orig_indices)}
    z_map = {orig_idx: int(nx_real + gari_idx) for gari_idx, orig_idx in enumerate(z_orig_indices)}
    
    if ordering_name == "order_1" or ordering_name == "order1":
        return real_x_gari + virt_x_gari + real_z_gari + virt_z_gari
    elif ordering_name == "order_2" or ordering_name == "order2":
        return real_x_gari + real_z_gari + virt_x_gari + virt_z_gari
    elif ordering_name == "order_3" or ordering_name == "order3":
        return real_x_gari + virt_z_gari + real_z_gari + virt_x_gari
    elif ordering_name == "order_4" or ordering_name == "order4":
        return real_x_gari + real_z_gari + virt_z_gari + virt_x_gari
    elif ordering_name == "order_8" or ordering_name == "order8":
        return real_x_gari + virt_x_gari + virt_z_gari + real_z_gari
    elif ordering_name == "order_9" or ordering_name == "order9":
        return real_z_gari + real_x_gari + virt_z_gari + virt_x_gari
    elif ordering_name == "order_10" or ordering_name == "order10":
        return real_z_gari + real_x_gari + virt_x_gari + virt_z_gari
        
    real_gari_chronological = []
    for orig_idx in range(num_original_detectors):
        if orig_idx in x_map:
            real_gari_chronological.append(x_map[orig_idx])
        elif orig_idx in z_map:
            real_gari_chronological.append(z_map[orig_idx])
            
    if ordering_name == "order_7" or ordering_name == "order7":
        virt_with_time_old = []
        for c in range(nx_virt):
            virt_with_time_old.append((time_vx_old[c], nx_real + nz_real + c))
        for c in range(len(time_vz_old)):
            virt_with_time_old.append((time_vz_old[c], nx_real + nz_real + nx_virt + c))
        virt_with_time_old.sort(key=lambda x: x[0])
        return real_gari_chronological + [v[1] for v in virt_with_time_old]
        
    elif ordering_name == "order_7a" or ordering_name == "order7a":
        virt_with_time_new = []
        for c in range(nx_virt):
            virt_with_time_new.append((time_vx_new[c], nx_real + nz_real + c))
        for c in range(len(time_vz_new)):
            virt_with_time_new.append((time_vz_new[c], nx_real + nz_real + nx_virt + c))
        virt_with_time_new.sort(key=lambda x: x[0])
        return real_gari_chronological + [v[1] for v in virt_with_time_new]
        
    elif ordering_name == "order_7b" or ordering_name == "order7b":
        virt_with_time_min_old = []
        for c in range(nx_virt):
            virt_with_time_min_old.append((time_vx_min_old[c], nx_real + nz_real + c))
        for c in range(len(time_vz_min_old)):
            virt_with_time_min_old.append((time_vz_min_old[c], nx_real + nz_real + nx_virt + c))
        virt_with_time_min_old.sort(key=lambda x: x[0])
        return real_gari_chronological + [v[1] for v in virt_with_time_min_old]
        
    elif ordering_name == "order_7c" or ordering_name == "order7c":
        virt_with_time_min_new = []
        for c in range(nx_virt):
            virt_with_time_min_new.append((time_vx_min_new[c], nx_real + nz_real + c))
        for c in range(len(time_vz_min_new)):
            virt_with_time_min_new.append((time_vz_min_new[c], nx_real + nz_real + nx_virt + c))
        virt_with_time_min_new.sort(key=lambda x: x[0])
        return real_gari_chronological + [v[1] for v in virt_with_time_min_new]
        
    def get_order_11_variant(v_x_times, v_z_times):
        all_det = []
        for orig_idx in range(num_original_detectors):
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
        
    if ordering_name == "order_11" or ordering_name == "order11":
        return get_order_11_variant(time_vx_new, time_vz_new)
    elif ordering_name == "order_11a" or ordering_name == "order11a":
        return get_order_11_variant(time_vx_old, time_vz_old)
    elif ordering_name == "order_11b" or ordering_name == "order11b":
        return get_order_11_variant(time_vx_min_new, time_vz_min_new)
        
    def get_order_12_variant(v_x_times, v_z_times):
        all_det = []
        for orig_idx in range(num_original_detectors):
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
        
    if ordering_name == "order_12" or ordering_name == "order12":
        return get_order_12_variant(time_vx_min_new, time_vz_min_new)
    elif ordering_name == "order_12a" or ordering_name == "order12a":
        return get_order_12_variant(time_vx_min_old, time_vz_min_old)
    elif ordering_name == "order_12b" or ordering_name == "order12b":
        return get_order_12_variant(time_vx_new, time_vz_new)

    raise ValueError(f"Unknown ordering {ordering_name}")


def get_two_stage_layout(
    gari_structure: dict, dem_file: str, top_prior_policy: str = "modeN"
) -> dict:
    """Describes the fixed GARI block layout using full DEM indices."""
    if top_prior_policy not in ("modeN", "modePR"):
        raise ValueError(f"Unsupported two-stage prior policy {top_prior_policy}")
    nx = int(gari_structure["nx_virt"])
    nz = int(gari_structure["nz_virt"])
    nx_real = int(gari_structure["nx_real"])
    nz_real = int(gari_structure["nz_real"])
    ny = int(len(gari_structure["i_hy"]))
    physical_detectors = int(gari_structure["num_original_detectors"])
    virtual_detectors = nx + nz

    if (
        gari_structure["dx"].shape != (nx_real, nx)
        or gari_structure["dz"].shape != (nz_real, nz)
        or nx_real + nz_real != physical_detectors
    ):
        raise ValueError("D_X and D_Z dimensions do not partition the GARI top block")

    block_counts = {
        "e_z": nx,
        "e_x": nz,
        "e_y": ny,
        "bar_e_z": nx,
        "bar_e_x": nz,
    }
    error_blocks = {}
    offset = 0
    for name, count in block_counts.items():
        error_blocks[name] = {"offset": offset, "count": count}
        offset += count

    physical_errors = nx + nz + ny
    barred_errors = virtual_detectors
    matrix = gari_structure["gari_matrix"]
    expected_shape = (physical_detectors + virtual_detectors, offset)
    if matrix.shape != expected_shape or physical_errors + barred_errors != offset:
        raise ValueError("GARI matrix dimensions do not agree with its block layout")

    # Physical errors belong entirely to the bottom virtual-detector problem.
    for error in range(physical_errors):
        rows = matrix.indices[matrix.indptr[error]:matrix.indptr[error + 1]]
        if np.any(rows < physical_detectors):
            raise ValueError(f"physical error {error} touches a physical detector")
        expected_weight = 1 if error < virtual_detectors else 2
        if len(rows) != expected_weight:
            raise ValueError(f"physical error {error} has invalid bottom degree")
        if error < virtual_detectors and rows.tolist() != [physical_detectors + error]:
            raise ValueError(f"physical error {error} misses its identity target")

    barred_to_virtual = []
    for local_index in range(barred_errors):
        error = physical_errors + local_index
        detector = physical_detectors + local_index
        rows = matrix.indices[matrix.indptr[error]:matrix.indptr[error + 1]]
        virtual_rows = rows[rows >= physical_detectors]
        if virtual_rows.tolist() != [detector]:
            raise ValueError(f"barred error {error} has an unexpected virtual target")
        barred_to_virtual.append({"error": error, "detector": detector})

    layout = {
        "schema": "gari_two_stage_layout",
        "version": 1,
        "dem_file": dem_file,
        "top_prior_policy": top_prior_policy,
        "bottom_prior_policy": (
            "original" if top_prior_policy == "modeN" else "lp_residual"
        ),
        "detectors": {
            "physical": {"offset": 0, "count": physical_detectors},
            "virtual": {"offset": physical_detectors, "count": virtual_detectors},
            "total_count": physical_detectors + virtual_detectors,
        },
        "errors": {
            "physical_count": physical_errors,
            "barred_count": barred_errors,
            "total_count": offset,
            "blocks": error_blocks,
        },
        "top_components": {
            "index_space": "monolithic_dem",
            "d_x": {
                "detector_rows": {"offset": 0, "count": nx_real},
                "barred_error_columns": {"offset": physical_errors, "count": nx},
                "debt_detector_rows": {"offset": physical_detectors, "count": nx},
            },
            "d_z": {
                "detector_rows": {"offset": nx_real, "count": nz_real},
                "barred_error_columns": {
                    "offset": physical_errors + nx,
                    "count": nz,
                },
                "debt_detector_rows": {
                    "offset": physical_detectors + nx,
                    "count": nz,
                },
            },
        },
        "barred_error_to_virtual_detector": barred_to_virtual,
    }
    if top_prior_policy == "modePR":
        layout.update({
            "prior_policy_id": "lp_floor_then_max_barred_v1",
            "final_cost_policy": "reconstructed_original_physical",
        })
    return layout


def process_directory(input_path, two_stage_only=False, two_stage_prior="modeN"):
    if two_stage_prior not in ("modeN", "modePR", "both"):
        raise ValueError(f"Unknown two-stage prior selection {two_stage_prior}")
    input_path = get_target_path(input_path)
    if os.path.isdir(input_path):
        stim_files = glob.glob(os.path.join(input_path, "**", "*.stim"), recursive=True)
    elif os.path.isfile(input_path):
        stim_files = [input_path] if input_path.endswith(".stim") else []
    else:
        if "*" not in input_path:
            input_path += "*"
        stim_files = [f for f in glob.glob(input_path, recursive=True) if f.endswith(".stim")]
    for stim_path in stim_files:
        print(f"Processing {stim_path}")
        try:
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
            
            gari_structure = gari_transform(H, L, det_types)
            
            gari_matrix = gari_structure["gari_matrix"]
            gari_obs_matrix = gari_structure["gari_obs_matrix"]
            gari_obs_matrix_og = gari_structure["gari_obs_matrix_og"]
            
            stim_dir = os.path.dirname(stim_path)
            stim_stem = os.path.splitext(os.path.basename(stim_path))[0]
            gari_dir = os.path.join(stim_dir, "gari")
            os.makedirs(gari_dir, exist_ok=True)
            base_path = os.path.join(gari_dir, stim_stem)
            
            # modes_to_generate = [
            #     "modeA", "modeB", "modeC", "modeF", "modeG", "modeH",
            #     "modeI", "modeJ", "modeK", "modeL", "modeM", "modeN",
            #     "modeO", "modeP", "modeQ", "modeR", "modeS", "modeU",
            #     "modeV", "modeS2", "modeSO", "modeSO2"
            # ]
            selected_two_stage_modes = {
                "modeN": ["modeN"],
                "modePR": ["modePR"],
                "both": ["modeN", "modePR"],
            }[two_stage_prior]
            if two_stage_only:
                modes_to_generate = selected_two_stage_modes
            else:
                modes_to_generate = [
                    "modeA", "modeN", "modeO", "modeQ", "modeR", "modeS",
                    "modeS2", "modeSO", "modeSO2"
                ]
                if "modePR" in selected_two_stage_modes:
                    modes_to_generate.append("modePR")
            
            for mode_name in modes_to_generate:
                try:
                    priors_for_mode = assign_prior_weights(gari_structure, mode_name, priors)
                    if not two_stage_only:
                        dem_for_mode = matrices_to_dem(
                            gari_matrix, gari_obs_matrix, priors_for_mode
                        )
                        dem_for_mode.to_file(base_path + f"_{mode_name}.dem")
                    
                    dem_for_mode_og = matrices_to_dem(gari_matrix, gari_obs_matrix_og, priors_for_mode)
                    dem_for_mode_og.to_file(base_path + f"_ogL_{mode_name}.dem")
                except NotImplementedError as e:
                    print(f"Skipping {mode_name}: {e}")
            
            x_orig_indices = np.where(det_types == 1)[0]
            z_orig_indices = np.where(det_types == 3)[0]
            
            nx = len(x_orig_indices)
            
            mapping = [-1] * dem.num_detectors
            for gari_idx, orig_idx in enumerate(x_orig_indices):
                mapping[int(orig_idx)] = int(gari_idx)
            for gari_idx, orig_idx in enumerate(z_orig_indices):
                mapping[int(orig_idx)] = int(nx + gari_idx)
                
            mapping_base = {
                "num_original_detectors": dem.num_detectors,
                "mapping": mapping,
            }
            if not two_stage_only:
                order_names = ("order2", "order4", "order7", "order9", "order10")
                mapping_base["det_orders"] = {
                    name: [
                        int(x)
                        for x in get_detector_orderings(gari_structure, det_types, name)
                    ]
                    for name in order_names
                }

            # Preserve the existing mode-N names. Mode PR uses a distinct
            # mapping so its residual physical weights cannot be mistaken for
            # the original physical weights stored by mode N.
            if not two_stage_only or "modeN" in selected_two_stage_modes:
                mode_n_mapping = dict(mapping_base)
                mode_n_mapping["gari_two_stage"] = get_two_stage_layout(
                    gari_structure, f"{stim_stem}_ogL_modeN.dem", "modeN"
                )
                mapping_suffix = (
                    "_two_stage_mapping.json" if two_stage_only else "_mapping.json"
                )
                with open(base_path + mapping_suffix, "w") as f:
                    json.dump(mode_n_mapping, f, indent=2)

            if "modePR" in selected_two_stage_modes:
                mode_pr_mapping = dict(mapping_base)
                mode_pr_mapping["gari_two_stage"] = get_two_stage_layout(
                    gari_structure, f"{stim_stem}_ogL_modePR.dem", "modePR"
                )
                with open(base_path + "_two_stage_modePR_mapping.json", "w") as f:
                    json.dump(mode_pr_mapping, f, indent=2)
                
            print(f"Successfully processed {stim_path}")
        except Exception as e:
            print(f"Error processing {stim_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process stim files and generate Gari DEMs.")
    parser.add_argument("input_path", type=str, help="Input directory, file prefix, or glob pattern for .stim files")
    parser.add_argument(
        "--two-stage-only",
        action="store_true",
        help="Generate only the selected physical-logical DEM and compact mapping",
    )
    parser.add_argument(
        "--two-stage-prior",
        choices=("modeN", "modePR", "both"),
        default="modeN",
        help=(
            "Compact prior policy (default: modeN). With full generation, "
            "modePR or both additionally generates mode PR artifacts."
        ),
    )
    args = parser.parse_args()
    process_directory(
        args.input_path,
        two_stage_only=args.two_stage_only,
        two_stage_prior=args.two_stage_prior,
    )
