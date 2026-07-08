import os
import json
import math
import re
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import binomtest
import numpy as np

def extract_circuit_info(circuit_path):
    info = { 'type': 'unknown', 'r': 1, 'd': 1, 'p': 0.0, 'q': 1 }
    if 'surfacecodes' in circuit_path: info['type'] = 'surfacecodes'
    elif 'colorcodes' in circuit_path: info['type'] = 'colorcodes'
    elif 'bivariatebicyclecodes' in circuit_path: info['type'] = 'bivariatebicyclecodes'
        
    m_r = re.search(r'r=(\d+)', circuit_path)
    if m_r: info['r'] = max(1, int(m_r.group(1)))
    m_d = re.search(r'd=(\d+)', circuit_path)
    if m_d: info['d'] = int(m_d.group(1))
    m_p = re.search(r'p=([\d\.]+)', circuit_path)
    if m_p: info['p'] = float(m_p.group(1))
    m_q = re.search(r'q=(\d+)', circuit_path)
    if m_q: info['q'] = int(m_q.group(1))
        
    return info

def process_data(filepath):
    data_groups = {}
    if not os.path.exists(filepath):
        print(f"Input file not found: {filepath}")
        return data_groups
        
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                row = json.loads(line)
            except:
                continue
                
            info = extract_circuit_info(row.get('circuit_path', ''))
            dem_path = row.get('dem_path', '')
            
            # Use 'unknown' for custom_order if it was omitted from the JSON (before the C++ fix)
            custom_order = row.get('custom_order', 'unknown')
            if custom_order == "": 
                custom_order = 'unknown'

            beam = row.get('det_beam', 0)
            num_det_orders = row.get('num_det_orders', 1)
            
            p_val, c_type, d_val, q_val, r_val = info['p'], info['type'], info['d'], info['q'], info['r']
            
            # Baselines don't provide a custom DEM file path via --dem
            is_baseline = (dem_path == "")
            
            # Determine operator type and mode
            operator_type = 'ogL' if '_ogL_' in dem_path else 'normal'
            mode = 'unknown'
            m_mode = re.search(r'_(mode[A-Za-z0-9]+)\.dem', dem_path)
            if m_mode:
                mode = m_mode.group(1)
            
            sparsify_errors = row.get('sparsify_errors', False)
            sparsify_reactivate_limit = row.get('sparsify_reactivate_limit', -1)
            beam_climbing = row.get('beam_climbing', True)
            
            key = (p_val, c_type, d_val, q_val, r_val, operator_type, mode, custom_order, beam, num_det_orders, is_baseline, sparsify_errors, sparsify_reactivate_limit, beam_climbing)
            
            if key not in data_groups:
                data_groups[key] = {
                    'total_time_seconds': 0.0, 'num_shots': 0, 'num_errors': 0, 
                    'num_low_confidence': 0
                }
                
            data_groups[key]['total_time_seconds'] += row.get('total_time_seconds', 0.0)
            data_groups[key]['num_shots'] += row.get('num_shots', 0)
            data_groups[key]['num_errors'] += row.get('num_errors', 0)
            data_groups[key]['num_low_confidence'] += row.get('num_low_confidence', 0)
            
    return data_groups

def compute_metrics(data_groups):
    metrics = []
    for key, agg in data_groups.items():
        p_val, c_type, d_val, q_val, r_val, op_type, mode, order, beam, num_det_orders, is_baseline, sparsify_errors, sparsify_reactivate_limit, beam_climbing = key
        shots = agg['num_shots']
        if shots == 0: continue
            
        errs = agg['num_errors'] + agg['num_low_confidence']
        time_sec = agg['total_time_seconds']
        
        p_raw = errs / shots
        try:
            ci = binomtest(k=errs, n=shots).proportion_ci(confidence_level=0.95)
            ler_err_low = (p_raw - ci.low) / r_val
            ler_err_high = (ci.high - p_raw) / r_val
        except:
            ler_err_low = 0
            ler_err_high = 0
            
        metrics.append({
            'p': p_val, 'type': c_type, 'd': d_val, 'q': q_val, 'r': r_val, 
            'op_type': op_type, 'mode': mode, 'order': order, 'beam': beam, 
            'num_det_orders': num_det_orders, 'is_baseline': is_baseline,
            'sparsify_errors': sparsify_errors, 'sparsify_reactivate_limit': sparsify_reactivate_limit,
            'beam_climbing': beam_climbing,
            'ler': p_raw / r_val,
            'ler_err_low': ler_err_low,
            'ler_err_high': ler_err_high,
            'time_per_round': time_sec / shots / r_val,
            'shots': shots,
            'num_low_confidence': agg['num_low_confidence']
        })
    return metrics

def format_gari_label(mode, order, beam, beam_climbing=True, sparsify_errors=False, sparsify_reactivate_limit=-1):
    m_str = mode.replace('mode', '')
    o_str = order.replace('order', '')
    beam_str = 'lb' if beam == 20 else f"b{beam}"
    label = f"{m_str}-{o_str}-{beam_str}"
    if not beam_climbing:
        label += "-no_bc"
    if sparsify_errors:
        label += "-sp"
        if sparsify_reactivate_limit != -1:
            label += f"({sparsify_reactivate_limit})"
    else:
        label += "-nosp"
    return label

def plot_ler_vs_time(metrics, p_filter, op_filter, c_type_filter, filename, title):
    plt.figure(figsize=(12, 9))
    filtered = [m for m in metrics if m['p'] == p_filter and m['op_type'] == op_filter and m['type'] == c_type_filter]
    # Baselines are shared between both normal and ogL plots
    baselines = [m for m in metrics if m['p'] == p_filter and m['is_baseline'] and m['type'] == c_type_filter]
    
    # Combine uniquely
    combined_metrics = {tuple(m.items()) for m in filtered + baselines}
    combined_metrics = [dict(t) for t in combined_metrics]
    
    if not combined_metrics:
        plt.close()
        return

    circuits = {}
    for m in combined_metrics:
        ckey = (m['type'], m['d'], m['q'])
        if ckey not in circuits: circuits[ckey] = []
        circuits[ckey].append(m)
        
    color_map = {'surfacecodes': '#5D95E8', 'colorcodes': '#F6C644', 'bivariatebicyclecodes': 'fuchsia'}
    base_color = color_map.get(c_type_filter, 'black')
    
    unique_qd = sorted(list(set((c[1], c[2]) for c in circuits.keys())))
    markers = ['o', 's', '^', 'D', 'p', 'h', 'X', '8', '>']
    marker_map = {unique_qd[i]: markers[i % len(markers)] for i in range(len(unique_qd))}
        
    for ckey, points in circuits.items():
        c_type, c_d, c_q = ckey
        marker = marker_map.get((c_d, c_q), 'o')
        
        baseline_pts = [p for p in points if p['is_baseline']]
        gari_pts = [p for p in points if not p['is_baseline']]
        
        # Plot GARI cloud
        for p in gari_pts:
            plt.scatter(p['time_per_round'], p['ler'], color=base_color, marker=marker, alpha=0.3, s=50, zorder=1)
            
        # Highlight Fastest and Most Accurate GARI for each beam size
        for b_val in [5, 20]:
            b_pts = [p for p in gari_pts if p['beam'] == b_val]
            if b_pts:
                fastest_gari = min(b_pts, key=lambda x: x['time_per_round'])
                most_acc_gari = min(b_pts, key=lambda x: x['ler'])
                
                if b_val == 5:
                    fc = 'white'
                    ec = base_color
                    lw = 2
                else:
                    fc = base_color
                    ec = 'black'
                    lw = 1.5

                # Fastest
                lbl_fast = format_gari_label(
                    fastest_gari['mode'], fastest_gari['order'], b_val,
                    fastest_gari.get('beam_climbing', True),
                    fastest_gari.get('sparsify_errors', False),
                    fastest_gari.get('sparsify_reactivate_limit', -1)
                )
                plt.scatter(fastest_gari['time_per_round'], fastest_gari['ler'], facecolor=fc, edgecolor=ec, 
                            marker='<', s=180, linewidths=lw, zorder=4)
                plt.errorbar(fastest_gari['time_per_round'], fastest_gari['ler'], yerr=[[fastest_gari['ler_err_low']], [fastest_gari['ler_err_high']]], 
                             fmt='none', ecolor=base_color, alpha=0.7, capsize=3, zorder=3)
                plt.annotate(lbl_fast, (fastest_gari['time_per_round'], fastest_gari['ler']), 
                             xytext=(-8, 5), textcoords='offset points', ha='right', va='bottom', fontsize=9, color=base_color)

                # Most Accurate
                if fastest_gari != most_acc_gari:
                    lbl_acc = format_gari_label(
                        most_acc_gari['mode'], most_acc_gari['order'], b_val,
                        most_acc_gari.get('beam_climbing', True),
                        most_acc_gari.get('sparsify_errors', False),
                        most_acc_gari.get('sparsify_reactivate_limit', -1)
                    )
                    plt.scatter(most_acc_gari['time_per_round'], most_acc_gari['ler'], facecolor=fc, edgecolor=ec, 
                                marker='v', s=180, linewidths=lw, zorder=4)
                    plt.errorbar(most_acc_gari['time_per_round'], most_acc_gari['ler'], yerr=[[most_acc_gari['ler_err_low']], [most_acc_gari['ler_err_high']]], 
                                 fmt='none', ecolor=base_color, alpha=0.7, capsize=3, zorder=3)
                    plt.annotate(lbl_acc, (most_acc_gari['time_per_round'], most_acc_gari['ler']), 
                                 xytext=(8, -5), textcoords='offset points', ha='left', va='top', fontsize=9, color=base_color)

        base_21 = None
        # Plot Baselines
        for p in baseline_pts:
            if p['num_det_orders'] == 1:
                plt.scatter(p['time_per_round'], p['ler'], facecolor='white', edgecolor=base_color, marker='*', s=400, linewidths=1.5, zorder=5)
                plt.errorbar(p['time_per_round'], p['ler'], yerr=[[p['ler_err_low']], [p['ler_err_high']]], fmt='none', ecolor=base_color, alpha=0.8, capsize=4, zorder=4)
            else:
                base_21 = p
                plt.scatter(p['time_per_round'], p['ler'], facecolor=base_color, edgecolor='black', marker='*', s=400, linewidths=1.5, zorder=5)
                plt.errorbar(p['time_per_round'], p['ler'], yerr=[[p['ler_err_low']], [p['ler_err_high']]], fmt='none', ecolor=base_color, alpha=0.8, capsize=4, zorder=4)

        # Draw connecting lines from the 21-order baseline to the ABSOLUTE fastest and ABSOLUTE most accurate GARI points only
        if base_21 and gari_pts:
            abs_fastest = min(gari_pts, key=lambda x: x['time_per_round'])
            abs_acc = min(gari_pts, key=lambda x: x['ler'])
            
            lines_to_draw = [abs_fastest]
            if abs_fastest != abs_acc:
                lines_to_draw.append(abs_acc)
                
            for h in lines_to_draw:
                x0, y0 = base_21['time_per_round'], base_21['ler']
                x1, y1 = h['time_per_round'], h['ler']
                
                # Plot the line
                plt.plot([x0, x1], [y0, y1], color=base_color, linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)
                
                # --- Geometric Midpoint Annotations ---
                speedup = x0 / x1 if x1 > 0 else 1
                
                # Calculate LER Uncertainty Bounds using log-relative-risk and Delta method
                k0 = y0 * base_21['r'] * base_21['shots']
                n0 = base_21['shots']
                k1 = y1 * h['r'] * h['shots']
                n1 = h['shots']
                
                k0_a, n0_a = k0 + 0.5, n0 + 0.5
                k1_a, n1_a = k1 + 0.5, n1 + 0.5
                
                r_adj = (k1_a / n1_a) / (k0_a / n0_a)
                se_log_r = math.sqrt(max(0, 1/k1_a - 1/n1_a + 1/k0_a - 1/n0_a))
                
                r_low = r_adj * math.exp(-1.96 * se_log_r)
                r_high = r_adj * math.exp(1.96 * se_log_r)
                
                if round(r_low, 2) == round(r_high, 2):
                    ler_str = f"{r_low:.2f}x err"
                else:
                    ler_str = f"{r_low:.2f}-{r_high:.2f}x err"
                
                mid_x = math.exp((math.log(x0) + math.log(x1)) / 2)
                mid_y = math.exp((math.log(y0) + math.log(y1)) / 2)
                
                plt.text(mid_x, mid_y * 1.05, f"{speedup:.1f}x spd\n{ler_str}", 
                         fontsize=7, color='black', ha='center', va='bottom', 
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8), zorder=6)

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.4) 
    
    valid_lers = [p['ler'] for p in combined_metrics if p['ler'] > 0]
    if valid_lers: plt.ylim(bottom=min(valid_lers) / 2.0) 
        
    plt.xlabel('Time per round (seconds)')
    plt.ylabel('Logical Error Rate per round')
    plt.title(title)
    
    # Custom Legend
    legend_elements = []
    
    # Distance Markers
    for (d_val, q_val) in unique_qd:
        mark = marker_map[(d_val, q_val)]
        label_str = f"d={d_val}" if c_type_filter != 'bivariatebicyclecodes' else f"d={d_val}, q={q_val}"
        legend_elements.append(mlines.Line2D([0], [0], color='none', marker=mark, markerfacecolor=base_color, markeredgecolor='none', markersize=10, label=label_str))
        
    legend_elements.append(mlines.Line2D([0], [0], color='none', label="")) # Spacer
    
    # Point types
    legend_elements.extend([
        mlines.Line2D([0], [0], color='none', marker='*', markerfacecolor='white', markeredgecolor='black', markersize=15, markeredgewidth=1.5, label='Baseline (longbeam, 1 order)'),
        mlines.Line2D([0], [0], color='none', marker='*', markerfacecolor='gray', markeredgecolor='black', markersize=15, markeredgewidth=1.5, label='Baseline (longbeam)'),
        mlines.Line2D([0], [0], color='none', marker='.', markerfacecolor='gray', markeredgecolor='none', markersize=15, alpha=0.4, label='All GARI configs (Cloud)'),
        mlines.Line2D([0], [0], color='none', marker='<', markerfacecolor='white', markeredgecolor='black', markersize=12, markeredgewidth=1.5, label='Fastest GARI (Beam 5)'),
        mlines.Line2D([0], [0], color='none', marker='v', markerfacecolor='white', markeredgecolor='black', markersize=12, markeredgewidth=1.5, label='Most Accurate GARI (Beam 5)'),
        mlines.Line2D([0], [0], color='none', marker='<', markerfacecolor='gray', markeredgecolor='black', markersize=12, markeredgewidth=1.5, label='Fastest GARI (Longbeam)'),
        mlines.Line2D([0], [0], color='none', marker='v', markerfacecolor='gray', markeredgecolor='black', markersize=12, markeredgewidth=1.5, label='Most Accurate GARI (Longbeam)'),
    ])
                                          
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, labelspacing=0.8)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.savefig(filename.replace('png', 'pdf'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    INPUT_FILE = 'aggregated_results.jsonl'
    OUTPUT_DIR = 'plots'
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        
    data_groups = process_data(INPUT_FILE)
    metrics = compute_metrics(data_groups)
    
    if len(metrics) > 0:
        print(f"Loaded {len(metrics)} aggregated data points. Generating plots...")
        
        # Dynamically find all error rates and code types in the JSON lines
        unique_p_vals = sorted(list(set(m['p'] for m in metrics)))
        unique_c_types = sorted(list(set(m['type'] for m in metrics)))
        display_names = {'surfacecodes': 'Surface Codes', 'colorcodes': 'Color Codes', 'bivariatebicyclecodes': 'Bicycle Codes'}
        
        for p_val in unique_p_vals:
            for c_type in unique_c_types:
                c_name = display_names.get(c_type, c_type)
                
                # Exclude 'normal' from filename per user request
                plot_ler_vs_time(
                    metrics, p_val, 'normal', c_type,
                    os.path.join(OUTPUT_DIR, f'ler_vs_time_{c_type}_p{p_val}.png'), 
                    f'{c_name} - LER vs Time (Operators in Auxiliary Columns, p={p_val})'
                )
                
                # Keep 'ogL' in filename per user request
                plot_ler_vs_time(
                    metrics, p_val, 'ogL', c_type,
                    os.path.join(OUTPUT_DIR, f'ler_vs_time_{c_type}_ogL_p{p_val}.png'), 
                    f'{c_name} - LER vs Time (Original Columns, p={p_val})'
                )
        print("Done!")
