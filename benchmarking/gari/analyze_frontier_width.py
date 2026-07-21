import argparse
import stim
import numpy as np
from scipy.sparse import coo_matrix, csgraph
import sys
import os
import glob

_src_py_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/py'))
if _src_py_dir not in sys.path:
    sys.path.insert(0, _src_py_dir)
from _tesseract_py_util.gari_dem_utils import dem_to_check_matrices

def get_target_path(relative_path):
    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY", "")
    return os.path.join(workspace_root, relative_path)

def find_file_path(codename, d, r, p, obs_basis, noise, suffix):
    r=d
    if codename == "surfacecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=surface_code_{obs_basis}"
    elif codename == "bivariatebicyclecodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=bivariate_bicycle_{obs_basis}"
    elif codename == "colorcodes":
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=superdense_color_code_{obs_basis}"
    else:
        fname = f"testdata/{codename}/r={r},d={d},p={p},noise={noise},c=UNKNOWN"

    fname = get_target_path(fname)
    # Find the definitive .stim file to grab the guaranteed absolute path
    stim_path = glob.glob(fname + '*.stim')[0]
    
    if suffix == '.stim' or not suffix:
        return stim_path
    elif suffix == '.dem':
        return stim_path[:-5] + suffix
    else:
        dirname = os.path.dirname(stim_path)
        basename = os.path.basename(stim_path)[:-5]
        return os.path.join(dirname, 'gari', basename + suffix)

def calculate_frontier_width(H):
    """
    Calculate the maximum frontier width of a sparse matrix H.
    H is (num_detectors, num_errors).
    For each row, finding the smallest and largest connected column index (s_i and e_i).
    The frontier width at column k is the number of rows i such that s_i <= k <= e_i.
    Returns the maximum frontier width over all columns.
    """
    H_csr = H.tocsr()
    num_rows, num_cols = H.shape
    s = np.full(num_rows, -1)
    e = np.full(num_rows, -1)
    for i in range(num_rows):
        indices = H_csr.indices[H_csr.indptr[i]:H_csr.indptr[i+1]]
        if len(indices) > 0:
            s[i] = np.min(indices)
            e[i] = np.max(indices)
    
    widths = np.zeros(num_cols, dtype=int)
    for k in range(num_cols):
        widths[k] = np.sum((s != -1) & (s <= k) & (k <= e))
    return widths

def main():
    parser = argparse.ArgumentParser(description="Analyze DEM frontier widths and RCM optimization.")
    parser.add_argument("--dem", type=str, help="Path to the Detector Error Model (.dem) file")
    parser.add_argument("--circuit", type=str, help="Path to the Circuit (.stim) file")
    parser.add_argument("--codename", type=str, help="e.g. surfacecodes, colorcodes")
    parser.add_argument("--obs-basis", type=str, default="Z")
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--p", type=float, default=0.001)
    parser.add_argument("--noise", type=str, default="si1000")
    parser.add_argument("--file-type", type=str, default="circuit", choices=["circuit", "dem"])
    parser.add_argument("--search-suffix", type=str, default=".stim", help="Suffix to append to the strict circuit path (e.g. '.dem' or '_modeN.dem')")
    args = parser.parse_args()

    if not args.dem and not args.circuit and not args.codename:
        print("No input provided. Falling back to default circuit...")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
        # args.circuit = os.path.join(repo_root, "testdata/surfacecodes/r=3,d=3,p=0.001,noise=si1000,c=surface_code_Z,q=17,gates=cz.stim")
        args.dem = os.path.join(repo_root, "testdata/surfacecodes/gari/r=5,d=5,p=0.001,noise=si1000,c=surface_code_Z,q=49,gates=cz_modeN.dem")

    if args.codename:
        resolved_path = find_file_path(args.codename, args.d, args.r, args.p, args.obs_basis, args.noise, args.search_suffix)
        print(f"Discovered file: {resolved_path}")
        if args.file_type == 'circuit':
            circuit = stim.Circuit.from_file(resolved_path)
            dem = circuit.detector_error_model(
                    decompose_errors=False,
                    flatten_loops=True,
                    ignore_decomposition_failures=True
                )
        elif args.file_type == 'dem':
            dem = stim.DetectorErrorModel.from_file(resolved_path)
    elif args.circuit:
        circuit = stim.Circuit.from_file(args.circuit)
        dem = circuit.detector_error_model(
                decompose_errors=False,
                flatten_loops=True,
                ignore_decomposition_failures=True
            )
    else:
        dem = stim.DetectorErrorModel.from_file(args.dem)

    # Use the robust extraction pipeline from GARI!
    H, L, priors, errors = dem_to_check_matrices(dem)
    
    seen = set()
    unique_cols = []
    # De-duplicate to build structural boundaries
    # for j, dets in enumerate(errors):
    #     dets_tuple = tuple(sorted(dets))
    #     if len(dets_tuple) > 0 and dets_tuple not in seen:
    #         seen.add(dets_tuple)
    #         unique_cols.append(j)
            
    # H = H.tocsc()[:, unique_cols]
    
    num_detectors = H.shape[0]
    num_faults = H.shape[1]

    widths_orig = calculate_frontier_width(H)

    print(f"Total Detectors: {num_detectors}")
    print(f"Total Error Mechanisms: {num_faults}")
    print(f"Original (Min, Max, Mean, Median): ({np.min(widths_orig)}, {np.max(widths_orig)}, {np.mean(widths_orig):.2f}, {np.median(widths_orig):.2f})")
    # print(f"Original Column-by-Column Widths: {list(widths_orig)}\n")

    # Adjacency graph of the columns
    HtH = H.T @ H
    
    # Reverse Cuthill-McKee
    perm = csgraph.reverse_cuthill_mckee(HtH)
    
    H_rcm = H.tocsc()[:, perm]
    
    widths_rcm = calculate_frontier_width(H_rcm)

    print(f"RCM Optimized (Min, Max, Mean, Median): ({np.min(widths_rcm)}, {np.max(widths_rcm)}, {np.mean(widths_rcm):.2f}, {np.median(widths_rcm):.2f})")
    # print(f"RCM Optimized Column-by-Column Widths: {list(widths_rcm)}")

if __name__ == "__main__":
    main()
