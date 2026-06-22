import stim
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import glob

def testdata_one_basis_circuit(circuit_og,z_basis=True):
    circuit = circuit_og.flattened().copy()
    for i in range(len(circuit)-1,-1,-1):
        if circuit[i].name == "DETECTOR":
            args = circuit[i].gate_args_copy()
            if z_basis and len(args) > 3 and args[3] <= 2:  
                # remove x detectors
                circuit.pop(i)

            if not z_basis and len(args) > 3 and args[3] >= 3:
                circuit.pop(i)

    return circuit

def load_circuit(codename,d,r,p,obs_basis,noise='si1000'):
    fname=""
    if codename == "surfacecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=surface_code_{obs_basis}"
    if codename == "bivariatebicyclecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=bivariate_bicycle_{obs_basis}"
    if codename == "colorcodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=superdense_color_code_{obs_basis}"
    circuit = None
    try:
        fname = glob.glob(fname + '*.stim')[0]
        circuit = stim.Circuit.from_file(fname)
        print("successful")
    except:
        raise("could not find the circuit")
    
    return circuit

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
    
    P_ez = P_dx
    P_ex = P_dz
    P_ey = P_y
    
    print("\n--- Physical Priors (From Stim) ---")
    print(f"P_ez (Hidden Z): min={np.min(P_ez):.2e}, max={np.max(P_ez):.2e}, mean={np.mean(P_ez):.2e}")
    print(f"P_ex (Hidden X): min={np.min(P_ex):.2e}, max={np.max(P_ex):.2e}, mean={np.mean(P_ex):.2e}")
    if len(P_ey) > 0:
        print(f"P_ey (Hidden Y): min={np.min(P_ey):.2e}, max={np.max(P_ey):.2e}, mean={np.mean(P_ey):.2e}")
    else:
        print("P_ey (Hidden Y): None")
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
    lam = 0.05
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
    
    time_vx = [np.max(hx_csc.indices[hx_csc.indptr[c]:hx_csc.indptr[c+1]]) if hx_csc.indptr[c+1] > hx_csc.indptr[c] else 0 for c in i_hx_only]
    time_vz = [np.max(hz_csc.indices[hz_csc.indptr[c]:hz_csc.indptr[c+1]]) if hz_csc.indptr[c+1] > hz_csc.indptr[c] else 0 for c in i_hz_only]
    
    if return_dem:
        return matrices_to_dem(gari_matrix, gari_obs_matrix, gari_priors_agg), nx, nz, time_vx, time_vz
    return gari_matrix, gari_obs_matrix, gari_priors_agg, gari_priors_keep_free, gari_priors_keep_scaled, gari_priors_hidden_free, gari_priors_tiny, gari_priors_hf_agg, gari_priors_tiny_agg, gari_priors_keep_keep, gari_priors_keep_max, gari_priors_hf_max, gari_priors_tiny_max, dx, dz, nx, nz, time_vx, time_vz


def set_prior_to_half(p_arr):
    # Sets the prior to 0.5, which translates to a weight of 0 in LLR domain
    return np.full_like(p_arr, 0.5)

def plot_gari_structure():
    print("Loading circuit d=3, r=3 for plotting...")
    circuit = load_circuit("surfacecodes", d=7, r=7, p=0.001, obs_basis='Z')
    dem = circuit.detector_error_model()
    
    H, L, priors, _ = dem_to_check_matrices(dem)
    det_types = get_detector_types(circuit)

    # Calculate ny
    hx = H[det_types == 0]
    hz = H[det_types == 1]
    hx_csc = hx.tocsc()
    hz_csc = hz.tocsc()
    col_nnz_x = np.diff(hx_csc.indptr)
    col_nnz_z = np.diff(hz_csc.indptr)
    i_hx_only = np.where((col_nnz_x > 0) & (col_nnz_z == 0))[0]
    i_hz_only = np.where((col_nnz_z > 0) & (col_nnz_x == 0))[0]
    i_both = np.where((col_nnz_x > 0) & (col_nnz_z > 0))[0]

    nx = len(i_hx_only)
    nz = len(i_hz_only)
    ny = len(i_both)
    ndx = hx.shape[0]
    ndz = hz.shape[0]

    print("Applying Gari Transform...")
    gari_matrix, _, _, _, _, _, _, _, _ = gari_transform(H, L, det_types, priors, return_dem=False)

    print("Plotting...")
    plt.figure(figsize=(10, 8))
    plt.spy(gari_matrix, markersize=1.5, color='blue')
    plt.title("Gari Matrix Block Structure (d=3, r=3)", fontsize=16)

    plt.tight_layout()
    
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gari_matrix_structure.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {out_path}")

def get_gari_orderings(dem, gari_dem, dx, dz, det_types, nx_virt):
    """
    Generates different detector processing orders for Tesseract's A* Search.
    
    Args:
        dem: The original Detector Error Model.
        gari_dem: The transformed Gari Detector Error Model.
        dx: The X-error physical dependency matrix.
        dz: The Z-error physical dependency matrix.
        det_types: Array indicating if a detector is X (1) or Z (3).
        nx_virt: Number of Virtual X detectors.
        
    Returns:
        A dictionary mapping ordering names to lists of detector indices.
    """
    nx_real = np.sum(det_types == 1)
    nz_real = np.sum(det_types == 3)
    
    # 1. Block-based Orderings (Process all of one type before moving to the next)
    real_x_gari = list(range(0, nx_real))
    real_z_gari = list(range(nx_real, nx_real + nz_real))
    virt_x_gari = list(range(nx_real + nz_real, nx_real + nz_real + nx_virt))
    virt_z_gari = list(range(nx_real + nz_real + nx_virt, gari_dem.num_detectors))
    
    # Extract original indices for fast lookup
    x_orig_indices = np.where(det_types == 1)[0]
    z_orig_indices = np.where(det_types == 3)[0]

    # Option 5: First-Touch Interleaved (Chronological)
    # Inserts virtual detectors immediately after the FIRST real detector they share an error with
    virtuals_to_insert_first = {i: [] for i in range(dem.num_detectors)}
    # Virtual X
    for c in range(dx.shape[1]):
        start = dx.indptr[c]
        end = dx.indptr[c+1]
        if start < end:
            min_orig_idx = min([x_orig_indices[r] for r in dx.indices[start:end]])
            virtuals_to_insert_first[min_orig_idx].append(nx_real + nz_real + c)
        else:
            virtuals_to_insert_first[0].append(nx_real + nz_real + c)
    # Virtual Z
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
    # Inserts virtual detectors immediately after the LAST real detector they share an error with
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


    # Option 7: Chronological Real, Chronological Virtual
    real_gari_chronological = []
    for orig_idx in range(dem.num_detectors):
        if orig_idx in x_orig_indices:
            gari_idx = np.where(x_orig_indices == orig_idx)[0][0]
            real_gari_chronological.append(int(gari_idx))
        elif orig_idx in z_orig_indices:
            gari_idx = np.where(z_orig_indices == orig_idx)[0][0]
            real_gari_chronological.append(int(nx_real + gari_idx))
            
    virt_with_time = []
    for c in range(dx.shape[1]):
        start = dx.indptr[c]
        end = dx.indptr[c+1]
        max_t = max([x_orig_indices[r] for r in dx.indices[start:end]]) if start < end else 0
        virt_with_time.append((max_t, nx_real + nz_real + c))
    for c in range(dz.shape[1]):
        start = dz.indptr[c]
        end = dz.indptr[c+1]
        max_t = max([z_orig_indices[r] for r in dz.indices[start:end]]) if start < end else 0
        virt_with_time.append((max_t, nx_real + nz_real + nx_virt + c))
        
    virt_with_time.sort(key=lambda x: x[0])
    virt_gari_chronological = [v[1] for v in virt_with_time]
    
    order_7 = real_gari_chronological + virt_gari_chronological

    return {
        # "Option 1  (RealX, VirtX, RealZ, VirtZ)": order_1,
        "Option 2  (RealX, RealZ, VirtX, VirtZ)": order_2,
        # "Option 3  (RealX, VirtZ, RealZ, VirtX)": order_3,
        "Option 4  (RealX, RealZ, VirtZ, VirtX)": order_4,
        # "Option 5  (First-Touch Interleaved)": order_5,
        # "Option 6  (Last-Touch Interleaved)": order_6,
        "Option 7  (Chrono Real, Chrono Virt)": order_7,
        # "Option 8  (RealX, VirtX, VirtZ, RealZ)": order_8,
    }

def test_gari_transform():
    print("Reading Circuit...")

    circuit = load_circuit("surfacecodes", d=9, r=9, p=0.001, obs_basis='Z')
    # circuit = load_circuit("bivariatebicyclecodes", d=12, r=12, p=0.002, obs_basis='Z')
    # circuit = load_circuit("colorcodes", d=11, r=11, p=0.001, obs_basis='Z')

    dem = circuit.detector_error_model()
    print("Extracting DEM...")
    dem = circuit.detector_error_model(
        decompose_errors=False, 
        flatten_loops=True, 
        ignore_decomposition_failures=True
    )
    # dem2 = circuit.detector_error_model(
    #     decompose_errors=True, 
    #     flatten_loops=True, 
    #     ignore_decomposition_failures=True
    # )
    # print(dem)
    # print("decomposed",dem2)
    det_types = get_detector_types(circuit)
    print(f"Total Detectors: {dem.num_detectors}")
    print(f"X Detectors: {np.sum(det_types == 1)}, Z Detectors: {np.sum(det_types == 3)}")
    
    print("Building original Check Matrix...")
    H, L, priors, errors = dem_to_check_matrices(dem)
    print(f"Original Check Matrix Shape: {H.shape}")
    print(f"Original Observables Matrix Shape: {L.shape}")
    
    # print("Applying Gari Transform (matrices)...")
    # H_gari, L_gari, priors_gari, L_ez_prime, L_ex_prime = gari_transform(H, L, det_types, priors, return_dem=False)
    # print(f"Gari Matrix Shape: {H_gari.shape}")
    # print(f"Gari Observables Matrix Shape: {L_gari.shape}")
    # print(f"Gari Priors Shape: {priors_gari.shape}")
    
    print("Applying Gari Transform (return_dem=False)...")
    gari_matrix, gari_obs_matrix, p_agg, p_keep_free, p_keep_scaled, p_hf, p_tiny, p_hf_agg, p_tiny_agg, p_keep_keep, p_keep_max, p_hf_max, p_tiny_max, dx, dz, nx_virt, nz_virt, time_vx, time_vz = gari_transform(H, L, det_types, priors, return_dem=False)
    
    dem_agg = matrices_to_dem(gari_matrix, gari_obs_matrix, p_agg)
    dem_keep_free = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_free)
    dem_keep_scaled = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_scaled)
    dem_hf = matrices_to_dem(gari_matrix, gari_obs_matrix, p_hf)
    dem_tiny = matrices_to_dem(gari_matrix, gari_obs_matrix, p_tiny)
    dem_hf_agg = matrices_to_dem(gari_matrix, gari_obs_matrix, p_hf_agg)
    dem_tiny_agg = matrices_to_dem(gari_matrix, gari_obs_matrix, p_tiny_agg)
    dem_keep_keep = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_keep)
    dem_keep_max = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_max)
    dem_hf_max = matrices_to_dem(gari_matrix, gari_obs_matrix, p_hf_max)
    dem_tiny_max = matrices_to_dem(gari_matrix, gari_obs_matrix, p_tiny_max)
    
    print(f"Gari Matrix Shape: {gari_matrix.shape}")
    print(f"Gari DEM has {dem_agg.num_detectors} detectors and {dem_agg.num_errors} errors.")
    
    orig_row_weights = np.sum(H.toarray(), axis=1)
    
    print(f"Original Avg Row Weight: {np.mean(orig_row_weights):.2f}")
    
    
    print("Success! Gari matrix generated and verified.")

    import time
    from tesseract_decoder.tesseract_sinter_compat import make_tesseract_sinter_decoders_dict
    import tesseract_decoder.tesseract as tesseract
    import tesseract_decoder.utils
    has_tesseract = True

    if has_tesseract:
        print("\n--- Running Tesseract Decoder Comparison ---")

        num_shots = 100
        sampler = circuit.compile_detector_sampler(seed=0)
        shots, obs_shots = sampler.sample(shots=num_shots, separate_observables=True)
        
        sinter_decoders = make_tesseract_sinter_decoders_dict()
        # short_beam_decoder_obj = sinter_decoders["tesseract-short-beam"]
        short_beam_decoder_obj = sinter_decoders["tesseract-long-beam"]
        base_config = short_beam_decoder_obj.compile_decoder_for_dem(dem=dem).decoder.config
        base_config.det_orders = [list(range(dem.num_detectors))]
        decoder_orig = tesseract.TesseractDecoder(base_config)
        # print(f"det_order_method: {short_beam_decoder_obj.det_order_method}")
        # print(f"seed: {short_beam_decoder_obj.seed}")
        
        print("\nCompiling Tesseract for Original DEM (Forced 1 Order)...")
        base_config.det_orders = [list(range(dem.num_detectors))]
        decoder_orig = tesseract.TesseractDecoder(base_config)
        
        print("Decoding Original DEM (1 Order)...")
        start_time = time.time()
        predicted_obs_orig = decoder_orig.decode_batch(shots)
        time_orig = time.time() - start_time
        
        correct_orig = np.sum(np.all(predicted_obs_orig == obs_shots, axis=1))
        print(f"Original DEM (1 Order): {correct_orig}/{num_shots} correct, Time: {time_orig:.4f}s")
        
        # We will test native orderings after gari_shots is defined!
        
        orders = get_gari_orderings(dem, dem_agg, dx, dz, det_types, nx_virt)
        
        # Pad and align shots for Gari DEM
        is_x_det = (det_types == 1)
        is_z_det = (det_types == 3)
        num_virtual = dem_agg.num_detectors - dem.num_detectors
        x_shots = shots[:, is_x_det]
        z_shots = shots[:, is_z_det]
        virtual_shots = np.zeros((num_shots, num_virtual), dtype=bool)
        gari_shots = np.concatenate([x_shots, z_shots, virtual_shots], axis=1)
        
        prior_modes = {
            # "Mode A: Aggregated (Original Double-Count)": dem_agg,
            # "Mode B: ez,ex,ey Keep, ez',ex' Free (0.499)": dem_keep_free,
            # "Mode C: ez,ex,ey Keep, ez',ex' Scaled (0.05x)": dem_keep_scaled,
            "Mode D: ez,ex,ey Free (0.499), ez',ex' Penalized": dem_hf,
            # "Mode E: ez,ex,ey Scaled (0.05x), ez',ex' Penalized": dem_tiny,
            "Mode F: ez,ex,ey Free (0.499), ez',ex' Aggregated": dem_hf_agg,
            "Mode G: ez,ex,ey Scaled (0.05x), ez',ex' Aggregated": dem_tiny_agg,
            # "Mode H: ez,ex,ey Keep, ez',ex' Keep": dem_keep_keep,
            # "Mode I: ez,ex,ey Keep, ez',ex' Maxed": dem_keep_max,
            "Mode J: ez,ex,ey Free, ez',ex' Maxed": dem_hf_max,
            # "Mode K: ez,ex,ey Scaled, ez',ex' Maxed": dem_tiny_max
        }
        
        print("\nTesting Priority Modes on Gari DEM ...")
        
        for mode_name, target_dem in prior_modes.items():
            print(f"\n>> {mode_name}")
            for name, order in orders.items():
                base_gari_config = short_beam_decoder_obj.compile_decoder_for_dem(dem=target_dem).decoder.config
                base_gari_config.det_orders = [order]
                base_gari_config.no_revisit_dets = False

                decoder_gari = tesseract.TesseractDecoder(base_gari_config)
                
                start_time = time.time()
                predicted_obs_gari = decoder_gari.decode_batch(gari_shots)
                time_gari = time.time() - start_time
                correct_gari = np.sum(np.all(predicted_obs_gari == obs_shots, axis=1))
                print(f"{name[:30]:30s} | {correct_gari}/{num_shots} correct | {time_gari:.4f}s")


if __name__ == "__main__":
    # plot_gari_structure()
    test_gari_transform()


