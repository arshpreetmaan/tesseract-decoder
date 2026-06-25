import os
import math
import json
from plot import process_data, compute_metrics, format_gari_label
from bokeh.plotting import figure, save, output_file
from bokeh.models import ColumnDataSource, CustomJS, CustomJSFilter, CDSView, HoverTool, TapTool, CheckboxGroup, Select, Button, Div
from bokeh.layouts import column, row

def generate_interactive_plot():
    # load metrics
    data_groups = process_data('aggregated_results.jsonl')
    metrics = compute_metrics(data_groups)

    # compute tradeoff data for all points
    baselines = {}
    for m in metrics:
        if m['is_baseline'] and m['num_det_orders'] > 1:
            key = (m['p'], m['type'], m['d'], m['q'], m['r'])
            baselines[key] = m

    source_data = {
        'x': [], 'y': [], 'color': [], 'marker': [], 'size': [], 'alpha': [], 'line_color': [], 'line_width': [],
        'p': [], 'type': [], 'd': [], 'q': [], 'r': [], 'op_type': [], 'mode': [], 'order': [], 'beam': [], 'is_baseline': [],
        'label': [], 'hover_label': [], 'ler_str': [], 'speedup_str': [], 'mid_x': [], 'mid_y': [], 'base_x': [], 'base_y': []
    }

    color_map = {'surfacecodes': '#5D95E8', 'colorcodes': '#F6C644', 'bivariatebicyclecodes': 'fuchsia'}
    markers = ['circle', 'square', 'triangle', 'diamond', 'hex', 'star', 'inverted_triangle']
    
    unique_qd = sorted(list(set((m['d'], m['q']) for m in metrics)))
    marker_map = {unique_qd[i]: markers[i % len(markers)] for i in range(len(unique_qd))}
    
    short_names = {'surfacecodes': 'sc', 'colorcodes': 'cc', 'bivariatebicyclecodes': 'bb'}
    type_display_names = {'surfacecodes': 'Surface Codes (sc)', 'colorcodes': 'Color Codes (cc)', 'bivariatebicyclecodes': 'Bivariate Bicycle Codes (bb)'}

    for m in metrics:
        # Omit Mode B and Mode C completely from plotting
        if m['mode'] in ['modeB', 'modeC']:
            continue
            
        x1, y1 = m['time_per_round'], m['ler']
        if y1 <= 0 or x1 <= 0:
            continue

        c = color_map.get(m['type'], 'black')
        mark = marker_map.get((m['d'], m['q']), 'circle')
        
        order_val = "Ensemble of all" if m['order'] == 'all' else m['order']

        source_data['x'].append(x1)
        source_data['y'].append(y1)
        source_data['color'].append(c)
        source_data['marker'].append(mark)

        if m['is_baseline']:
            source_data['line_color'].append('black')
            source_data['line_width'].append(2)
            source_data['alpha'].append(1.0)
            if m['num_det_orders'] == 1:
                source_data['size'].append(15)
            else:
                source_data['size'].append(20)
        else:
            source_data['line_color'].append(c)
            source_data['line_width'].append(0)
            source_data['size'].append(10)
            source_data['alpha'].append(0.5)

        source_data['p'].append(str(m['p']))
        source_data['type'].append(m['type'])
        source_data['d'].append(str(m['d']))
        source_data['q'].append(str(m['q']))
        source_data['r'].append(str(m['r']))
        source_data['op_type'].append('Auxiliary Columns' if m['op_type'] == 'normal' else 'Original Columns')
        source_data['mode'].append(m['mode'])
        source_data['order'].append(order_val)
        source_data['beam'].append(str(m['beam']) if m['beam'] != 20 else 'longbeam')
        source_data['is_baseline'].append(str(m['is_baseline']))

        label = "Baseline" if m['is_baseline'] else format_gari_label(m['mode'], m['order'], m['beam'])
        source_data['label'].append(label)
        
        short = short_names.get(m['type'], m['type'])
        code_dist_str = f"{short}, d={m['d']}"
        if m['is_baseline']:
            hover_label = f"Baseline [{code_dist_str}]"
        else:
            hover_label = f"{label} [{code_dist_str}]"
        source_data['hover_label'].append(hover_label)

        base_key = (m['p'], m['type'], m['d'], m['q'], m['r'])
        if not m['is_baseline'] and base_key in baselines:
            base_21 = baselines[base_key]
            x0, y0 = base_21['time_per_round'], base_21['ler']
            
            speedup = x0 / x1 if x1 > 0 else 1

            k0 = y0 * base_21['r'] * base_21['shots']
            n0 = base_21['shots']
            k1 = y1 * m['r'] * m['shots']
            n1 = m['shots']

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

            mid_x = math.exp((math.log(x0) + math.log(x1)) / 2) if x0>0 and x1>0 else (x0+x1)/2
            mid_y = math.exp((math.log(y0) + math.log(y1)) / 2) if y0>0 and y1>0 else (y0+y1)/2

            source_data['speedup_str'].append(f"{speedup:.1f}x spd")
            source_data['ler_str'].append(ler_str)
            source_data['mid_x'].append(mid_x)
            source_data['mid_y'].append(mid_y)
            source_data['base_x'].append(x0)
            source_data['base_y'].append(y0)
        else:
            source_data['speedup_str'].append("")
            source_data['ler_str'].append("")
            source_data['mid_x'].append(x1)
            source_data['mid_y'].append(y1)
            source_data['base_x'].append(x1)
            source_data['base_y'].append(y1)

    source = ColumnDataSource(data=source_data)

    line_source = ColumnDataSource(data={'x': [], 'y': []})
    text_source = ColumnDataSource(data={'x': [], 'y': [], 'text': []})

    unique_types = sorted(list(set(source_data['type'])))
    unique_modes = sorted(list(set(m for m in source_data['mode'] if m != 'unknown')))
    unique_orders = sorted(list(set(o for o in source_data['order'] if o != 'unknown')))
    unique_op_types = sorted(list(set(source_data['op_type'])))
    unique_beams = sorted(list(set(b for b in source_data['beam'] if b != '0')))
    unique_p = sorted(list(set(source_data['p'])))

    # Native Bokeh UI Controls for Distance
    code_sections = []
    d_chk_widgets = {}
    d_chk_keys = []

    for ct in unique_types:
        d_list = sorted(list(set(source_data['d'][i] for i in range(len(source_data['d'])) if source_data['type'][i] == ct)))
        if not d_list:
            continue
        
        name = type_display_names.get(ct, ct)
        btn = Button(label=f"{name} ▼", button_type="default", width=330, height=30)
        
        d_rows = []
        for d in d_list:
            td_val = f"{ct}_{d}"
            d_chk_keys.append(td_val)
            
            c = color_map.get(ct, 'black')
            
            # find marker
            mark = 'circle'
            for i in range(len(source_data['type'])):
                if source_data['type'][i] == ct and source_data['d'][i] == d:
                    mark = source_data['marker'][i]
                    break
            
            sym = '●'
            if mark == 'square': sym = '■'
            elif mark == 'triangle': sym = '▲'
            elif mark == 'diamond': sym = '◆'
            elif mark == 'hex': sym = '⬢'
            elif mark == 'star': sym = '★'
            elif mark == 'inverted_triangle': sym = '▼'
            
            chk = CheckboxGroup(labels=[f"Distance {d}"], active=[0], width=100)
            mark_div = Div(text=f"<div style='color:{c}; font-size:16px; margin-top:-2px;'>{sym}</div>", width=30)
            
            d_chk_widgets[td_val] = chk
            d_rows.append(row(chk, mark_div, sizing_mode="fixed", width=150, height=30))
            
        d_col = column(*d_rows, visible=False)
        btn.js_on_click(CustomJS(args=dict(col=d_col), code="col.visible = !col.visible;"))
        code_sections.append(column(btn, d_col))

    accordion_container = column(*code_sections)

    # General UI Controls
    mode_group = CheckboxGroup(labels=unique_modes, active=list(range(len(unique_modes))))
    order_group = CheckboxGroup(labels=unique_orders, active=list(range(len(unique_orders))))
    op_type_group = CheckboxGroup(labels=unique_op_types, active=list(range(len(unique_op_types))))
    beam_group = CheckboxGroup(labels=unique_beams, active=list(range(len(unique_beams))))
    p_select = Select(title="Physical Error Rate (p)", value=unique_p[0] if unique_p else "", options=unique_p)

    mode_select_all = Button(label="Select All", button_type="success", width=70, height=30, sizing_mode="fixed")
    mode_clear_all = Button(label="Clear All", button_type="primary", width=70, height=30, sizing_mode="fixed")
    order_select_all = Button(label="Select All", button_type="success", width=70, height=30, sizing_mode="fixed")
    order_clear_all = Button(label="Clear All", button_type="primary", width=70, height=30, sizing_mode="fixed")

    filter_args = dict(
        mode_group=mode_group, order_group=order_group,
        op_type_group=op_type_group, beam_group=beam_group, p_select=p_select,
        unique_modes=unique_modes, unique_orders=unique_orders,
        unique_op_types=unique_op_types, unique_beams=unique_beams,
        d_chk_keys=d_chk_keys, d_chk_widgets=d_chk_widgets
    )

    js_code = """
        const indices = [];
        const active_modes = mode_group.active.map(i => unique_modes[i]);
        const active_orders = order_group.active.map(i => unique_orders[i]);
        const active_op_types = op_type_group.active.map(i => unique_op_types[i]);
        const active_beams = beam_group.active.map(i => unique_beams[i]);
        const active_p = p_select.value;
        
        let active_td = [];
        for (let i = 0; i < d_chk_keys.length; i++) {
            let key = d_chk_keys[i];
            let chk = d_chk_widgets[key];
            if (chk.active.includes(0)) {
                active_td.push(key);
            }
        }

        for (let i = 0; i < source.get_length(); i++) {
            if (source.data['p'][i] !== active_p) continue;
            
            const t = source.data['type'][i];
            const d = source.data['d'][i];
            const td_key = t + "_" + d;
            
            if (!active_td.includes(td_key)) continue;
            
            const is_base = source.data['is_baseline'][i] === 'True';
            if (is_base) {
                indices.push(i);
            } else {
                if (active_modes.includes(source.data['mode'][i]) &&
                    active_orders.includes(source.data['order'][i]) &&
                    active_op_types.includes(source.data['op_type'][i]) &&
                    active_beams.includes(source.data['beam'][i])) {
                    indices.push(i);
                }
            }
        }
        return indices;
    """

    js_filter = CustomJSFilter(args=filter_args, code=js_code)
    view = CDSView(filter=js_filter)

    fastest_data = {'x': [], 'y': [], 'marker': []}
    accurate_data = {'x': [], 'y': [], 'marker': []}
    best_x = {}
    best_y = {}
    active_p = unique_p[0] if unique_p else ""
    for i in range(len(source_data['x'])):
        if source_data['p'][i] != active_p:
            continue
        if source_data['is_baseline'][i] == 'True':
            continue
        
        t = source_data['type'][i]
        d = source_data['d'][i]
        key = f"{t}_{d}"
        
        if key not in best_x:
            best_x[key] = {'min_x': float('inf'), 'x': -1, 'y': -1, 'marker': 'circle'}
            best_y[key] = {'min_y': float('inf'), 'x': -1, 'y': -1, 'marker': 'circle'}
            
        x, y = source_data['x'][i], source_data['y'][i]
        if x < best_x[key]['min_x']:
            best_x[key]['min_x'] = x
            best_x[key]['x'] = x
            best_x[key]['y'] = y
            best_x[key]['marker'] = source_data['marker'][i]
            
        if y < best_y[key]['min_y']:
            best_y[key]['min_y'] = y
            best_y[key]['x'] = x
            best_y[key]['y'] = y
            best_y[key]['marker'] = source_data['marker'][i]

    for k in best_x:
        if best_x[k]['x'] != -1:
            fastest_data['x'].append(best_x[k]['x'])
            fastest_data['y'].append(best_x[k]['y'])
            fastest_data['marker'].append(best_x[k]['marker'])
    for k in best_y:
        if best_y[k]['x'] != -1:
            accurate_data['x'].append(best_y[k]['x'])
            accurate_data['y'].append(best_y[k]['y'])
            accurate_data['marker'].append(best_y[k]['marker'])
    
    fastest_source = ColumnDataSource(data=fastest_data)
    accurate_source = ColumnDataSource(data=accurate_data)

    update_args = dict(
        source=source, fastest_source=fastest_source, accurate_source=accurate_source,
        mode_group=mode_group, order_group=order_group,
        op_type_group=op_type_group, beam_group=beam_group, p_select=p_select,
        unique_modes=unique_modes, unique_orders=unique_orders,
        unique_op_types=unique_op_types, unique_beams=unique_beams,
        d_chk_keys=d_chk_keys, d_chk_widgets=d_chk_widgets
    )

    update_js_code = """
        const active_modes = mode_group.active.map(i => unique_modes[i]);
        const active_orders = order_group.active.map(i => unique_orders[i]);
        const active_op_types = op_type_group.active.map(i => unique_op_types[i]);
        const active_beams = beam_group.active.map(i => unique_beams[i]);
        const active_p = p_select.value;
        
        let active_td = [];
        for (let i = 0; i < d_chk_keys.length; i++) {
            let key = d_chk_keys[i];
            let chk = d_chk_widgets[key];
            if (chk.active.includes(0)) {
                active_td.push(key);
            }
        }

        let best_x = {};
        let best_y = {};

        for (let i = 0; i < source.get_length(); i++) {
            if (source.data['p'][i] !== active_p) continue;
            
            const t = source.data['type'][i];
            const d = source.data['d'][i];
            const key = t + "_" + d;
            
            if (!active_td.includes(key)) continue;
            
            const is_base = source.data['is_baseline'][i] === 'True';
            if (!is_base) {
                if (active_modes.includes(source.data['mode'][i]) &&
                    active_orders.includes(source.data['order'][i]) &&
                    active_op_types.includes(source.data['op_type'][i]) &&
                    active_beams.includes(source.data['beam'][i])) {
                    
                    const x = source.data['x'][i];
                    const y = source.data['y'][i];
                    
                    if (!(key in best_x)) {
                        best_x[key] = {min_x: Infinity, x: -1, y: -1, marker: 'circle'};
                        best_y[key] = {min_y: Infinity, x: -1, y: -1, marker: 'circle'};
                    }
                    if (x < best_x[key].min_x) {
                        best_x[key].min_x = x; best_x[key].x = x; best_x[key].y = y; best_x[key].marker = source.data['marker'][i];
                    }
                    if (y < best_y[key].min_y) {
                        best_y[key].min_y = y; best_y[key].x = x; best_y[key].y = y; best_y[key].marker = source.data['marker'][i];
                    }
                }
            }
        }
        
        const f_x = [], f_y = [], f_marker = [];
        const a_x = [], a_y = [], a_marker = [];
        
        for (const k in best_x) {
            if (best_x[k].x !== -1) {
                f_x.push(best_x[k].x); f_y.push(best_x[k].y); f_marker.push(best_x[k].marker);
            }
        }
        for (const k in best_y) {
            if (best_y[k].x !== -1) {
                a_x.push(best_y[k].x); a_y.push(best_y[k].y); a_marker.push(best_y[k].marker);
            }
        }
        
        fastest_source.data = {'x': f_x, 'y': f_y, 'marker': f_marker};
        accurate_source.data = {'x': a_x, 'y': a_y, 'marker': a_marker};
        fastest_source.change.emit();
        accurate_source.change.emit();

        source.change.emit();
    """

    update_js = CustomJS(args=update_args, code=update_js_code)

    for ctrl in [mode_group, order_group, op_type_group, beam_group]:
        ctrl.js_on_change('active', update_js)
    p_select.js_on_change('value', update_js)
    
    # bind all distance checkboxes natively
    for td_val, chk in d_chk_widgets.items():
        chk.js_on_change('active', update_js)

    mode_select_all.js_on_click(CustomJS(args=dict(group=mode_group), code="group.active = Array.from({length: group.labels.length}, (v, k) => k);"))
    mode_clear_all.js_on_click(CustomJS(args=dict(group=mode_group), code="group.active = [];"))
    order_select_all.js_on_click(CustomJS(args=dict(group=order_group), code="group.active = Array.from({length: group.labels.length}, (v, k) => k);"))
    order_clear_all.js_on_click(CustomJS(args=dict(group=order_group), code="group.active = [];"))

    p = figure(width=1000, height=800, title="GARI Tradeoffs", 
               x_axis_type="log", y_axis_type="log",
               x_axis_label="Time per round (seconds)", y_axis_label="Logical Error Rate per round",
               tools="pan,wheel_zoom,box_zoom,reset,tap,save")

    scatter = p.scatter(x='x', y='y', source=source, view=view, size='size', color='color', line_color='line_color', line_width='line_width', marker='marker', alpha='alpha',
                        nonselection_alpha=0.1)

    p.scatter(x='x', y='y', source=fastest_source, marker='marker', size=20, fill_color=None, line_color='green', line_width=4, alpha=1.0)
    p.scatter(x='x', y='y', source=accurate_source, marker='marker', size=20, fill_color=None, line_color='red', line_width=4, alpha=1.0)

    p.line(x='x', y='y', source=line_source, color="black", line_dash="dashed", line_width=2, alpha=0.6)
    p.text(x='x', y='y', text='text', source=text_source, text_font_size="9pt",
           x_offset=5, y_offset=5, text_baseline="bottom")

    hover = HoverTool(renderers=[scatter], tooltips=[
        ("Config", "@hover_label"),
        ("Time", "@x"),
        ("LER", "@y")
    ])
    p.add_tools(hover)

    tap_js = CustomJS(args=dict(source=source, line_source=line_source, text_source=text_source), code="""
        const sel = source.selected.indices;
        if (sel.length > 0) {
            const i = sel[0];
            const is_base = source.data['is_baseline'][i] === 'True';
            if (!is_base && source.data['speedup_str'][i] !== "") {
                const x0 = source.data['base_x'][i];
                const y0 = source.data['base_y'][i];
                const x1 = source.data['x'][i];
                const y1 = source.data['y'][i];
                
                line_source.data = {'x': [x0, x1], 'y': [y0, y1]};
                line_source.change.emit();
                
                const mid_x = source.data['mid_x'][i];
                const mid_y = source.data['mid_y'][i];
                const spd = source.data['speedup_str'][i];
                const ler = source.data['ler_str'][i];
                
                text_source.data = {'x': [mid_x], 'y': [mid_y], 'text': [spd + "\\n" + ler]};
                text_source.change.emit();
                return;
            }
        }
        line_source.data = {'x': [], 'y': []};
        line_source.change.emit();
        text_source.data = {'x': [], 'y': [], 'text': []};
        text_source.change.emit();
    """)
    
    tap = p.select(type=TapTool)
    source.selected.js_on_change('indices', tap_js)

    controls = column(
        p_select,
        Div(text="<b>Code Types & Distances</b>"),
        accordion_container,
        Div(text="<b>Prior Modes</b>"),
        row(mode_select_all, mode_clear_all),
        mode_group,
        Div(text="<b>Detector Orders</b>"),
        row(order_select_all, order_clear_all),
        order_group,
        Div(text="<b>Operation Types</b>"),
        op_type_group,
        Div(text="<b>Beams</b>"),
        beam_group,
        width=350
    )

    MODE_DESC = {
        'modeA': "ez,ex,ey Original (Keep), ez',ex' Aggregated",
        'modeD': "ez,ex,ey Free, ez',ex' Penalized",
        'modeE': "ez,ex,ey Scaled, ez',ex' Penalized",
        'modeF': "ez,ex,ey Free, ez',ex' Aggregated",
        'modeG': "ez,ex,ey Scaled to make the original columns almost free cost, ez',ex' Aggregated",
        'modeH': "ez,ex,ey Keep, ez',ex' Keep",
        'modeI': "ez,ex,ey Keep, ez',ex' Maxed",
        'modeJ': "ez,ex,ey Free, ez',ex' Maxed",
        'modeK': "ez,ex,ey Scaled, ez',ex' Maxed",
        'modeL': "ez,ex Keep, ey Free, ez',ex' Aggregated",
        'modeM': "ez,ex Keep, ey Free, ez',ex' Maxed",
        'modeN': "ez,ex,ey Keep, ez',ex' XOR Aggregated",
        'modeO': "ez,ex,ey Scaled, ez',ex' XOR Aggregated"
    }

    ORDER_DESC = {
        'order1': "RealX, VirtX, RealZ, VirtZ",
        'order2': "RealX, RealZ, VirtX, VirtZ",
        'order3': "RealX, VirtZ, RealZ, VirtX",
        'order4': "RealX, RealZ, VirtZ, VirtX",
        'order5': "First-Touch Interleaved Chrono",
        'order6': "Last-Touch Interleaved Chrono",
        'order7': "Chrono Real, Chrono Virt",
        'order7a': "Chrono Real, Chrono Virt (With ey max)",
        'order7b': "Chrono Real, Chrono Virt min (No ey)",
        'order7c': "Chrono Real, Chrono Virt min (With ey)",
        'order8': "RealX, VirtX, VirtZ, RealZ",
        'order9': "RealZ, RealX, VirtZ, VirtX",
        'order10': "RealZ, RealX, VirtX, VirtZ",
        'Ensemble of all': "A combination of all multiple detector orders"
    }

    modes_html = ""
    for m in unique_modes:
        desc = MODE_DESC.get(m, "Unknown Mode")
        modes_html += f"<li><b>{m}:</b> {desc}</li>\n"

    orders_html = ""
    for o in unique_orders:
        desc = ORDER_DESC.get(o, "Unknown Order")
        orders_html += f"<li><b>{o}:</b> {desc}</li>\n"

    manual_div_str = f"""
        <div style='margin-top: 20px; font-size: 14px; max-width: 1000px;'>
            <b>Legend & Manual</b><br>
            <ul>
                <li><b>Baseline:</b> <span style='border:2px solid black; padding: 0 4px;'>Black Border</span> (Default: Longbeam, 1 Order, Normal Op).</li>
                <li><b>Highlights:</b> <span style='border:2px solid green; padding: 0 4px;'>Green Border</span> = Fastest Gari (Current View). <span style='border:2px solid red; padding: 0 4px;'>Red Border</span> = Most Accurate Gari (Current View).</li>
                <li><b>Gari Simulations:</b> Uses 1 detector order and <code>no-det-revisit=False</code> for A* Search.</li>
            </ul>
            <div style='display: flex; gap: 40px;'>
                <div>
                    <b>Prior Modes:</b>
                    <ul style='margin-top: 5px;'>
                        {modes_html}
                    </ul>
                </div>
                <div>
                    <b>Detector Orders:</b>
                    <ul style='margin-top: 5px;'>
                        {orders_html}
                    </ul>
                </div>
            </div>
        </div>
    """
    manual_div = Div(text=manual_div_str)

    layout = column(row(controls, p), manual_div)

    output_dir = 'plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file(os.path.join(output_dir, "interactive_plot.html"), title="GARI Interactive Plot")
    save(layout)

if __name__ == '__main__':
    generate_interactive_plot()
    print("Successfully generated plots/interactive_plot.html")
