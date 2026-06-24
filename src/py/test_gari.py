import stim
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
import glob
from gari_dem_utils import get_detector_types, dem_to_check_matrices, matrices_to_dem, gari_transform, get_gari_orderings

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



def test_gari_transform():
    print("Reading Circuit...")
    import sys
    if len(sys.argv) > 1:
        circuit = stim.Circuit.from_file(sys.argv[1])
    else:
        circuit = load_circuit("surfacecodes", d=5, r=5, p=0.001, obs_basis='Z')
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
    res = gari_transform(H, L, det_types, priors, return_dem=False)
    gari_matrix, gari_obs_matrix = res[0], res[1]
    p_agg, p_keep_free, p_keep_scaled, p_hf, p_tiny = res[2], res[3], res[4], res[5], res[6]
    p_hf_agg, p_tiny_agg, p_keep_keep, p_keep_max = res[7], res[8], res[9], res[10]
    p_hf_max, p_tiny_max, p_keep_freeY_agg, p_keep_freeY_max, p_keep_xor, p_scaled_xor = res[11], res[12], res[13], res[14], res[15], res[16]
    dx, dz, nx_virt, nz_virt, time_vx, time_vz = res[17], res[18], res[19], res[20], res[21], res[22]
    gari_obs_matrix_og = res[23]
    
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
    dem_keep_freeY_agg = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_freeY_agg)
    dem_keep_freeY_max = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_freeY_max)
    dem_hf_xor = matrices_to_dem(gari_matrix, gari_obs_matrix, p_keep_xor)
    dem_scaled_xor = matrices_to_dem(gari_matrix, gari_obs_matrix, p_scaled_xor)
    
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

        num_shots = 100000
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
        
        orders = get_gari_orderings(dem, dem_agg, res[17], res[18], det_types, res[19], res[21], res[22], res[24], res[25], res[26], res[27], res[28], res[29])
        
        # Pad and align shots for Gari DEM
        is_x_det = (det_types == 1)
        is_z_det = (det_types == 3)
        num_virtual = dem_agg.num_detectors - dem.num_detectors
        x_shots = shots[:, is_x_det]
        z_shots = shots[:, is_z_det]
        virtual_shots = np.zeros((num_shots, num_virtual), dtype=bool)
        gari_shots = np.concatenate([x_shots, z_shots, virtual_shots], axis=1)
        
        prior_modes = {
            "Mode A: Aggregated (Original Double-Count)": dem_agg,
            # "Mode B: ez,ex,ey Keep, ez',ex' Free (0.499)": dem_keep_free,
            # "Mode C: ez,ex,ey Keep, ez',ex' Scaled (0.05x)": dem_keep_scaled,
            # "Mode D: ez,ex,ey Free (0.499), ez',ex' Penalized": dem_hf,
            # "Mode E: ez,ex,ey Scaled (0.05x), ez',ex' Penalized": dem_tiny,
            # "Mode F: ez,ex,ey Free (0.499), ez',ex' Aggregated": dem_hf_agg,
            # "Mode G: ez,ex,ey Scaled (0.05x), ez',ex' Aggregated": dem_tiny_agg,
            # "Mode H: ez,ex,ey Keep, ez',ex' Keep": dem_keep_keep,
            "Mode I: ez,ex,ey Keep, ez',ex' Maxed": dem_keep_max,
            # "Mode J: ez,ex,ey Free, ez',ex' Maxed": dem_hf_max,
            # "Mode K: ez,ex,ey Scaled, ez',ex' Maxed": dem_tiny_max
            "Mode N: ez,ex,ey Keep, ez',ex' XOR Aggregated": dem_hf_xor,
            "Mode O: ez,ex,ey Scaled, ez',ex' XOR Aggregated": dem_scaled_xor
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


