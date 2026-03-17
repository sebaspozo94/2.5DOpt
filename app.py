import streamlit as st
import numpy as np
import logic  # Ensure logic.py/logic.so is in the same directory
import plotly.graph_objects as go
import io
import stl              
from stl import mesh 
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from skimage import measure  # For 3D Volumetric Meshing

st.set_page_config(page_title="2.5D Topology Opt", layout="centered")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 0rem; }
    .tag-container { display: flex; gap: 10px; margin-bottom: 1.5rem; }
    .tag { background-color: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 9999px; font-size: 0.85rem; font-weight: 500; border: 1px solid #e2e8f0; }
    .section-header { font-size: 1.25rem; font-weight: 700; color: #1e293b; margin-top: 1rem; margin-bottom: 0.5rem; }
    /* Forces the updating image to match the Plotly charts */
    [data-testid="stImage"] img {
        max-height: 600px;
        width: auto;
        object-fit: contain;
        margin-left: auto;
        margin-right: auto;
        display: block;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# PART 1: HEADER & OBJECTIVE
# ==========================================
st.markdown('<div class="main-header">2.5D Topology Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="tag-container"><span class="tag">2.5D</span><span class="tag">Plane Stress</span><span class="tag">Optimization</span><span class="tag">FEA Engine</span></div>', unsafe_allow_html=True)

with st.expander("🎯 App Objective", expanded=False):
    st.markdown("""
    **Objective:** Distribute a constant amount of material to maximize the stiffness of a 2.5D membrane structure 
    subjected to in-plane point loads.
    * The amount of material distributed is given by a fraction of the solid with sides: Domain X, Domain y, Min Thickness.
    
    📄 [**Read the detailed code explanation and documentation here (PDF)**](https://github.com/sebaspozo94/2.5DOpt/blob/main/2.5DOpt.pdf)
    """)

# --- SETUP SESSION STATE ---
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
    st.session_state.history = None
    st.session_state.X = None
    st.session_state.Y = None

# Base is fixed at (0,0) and (4,0) with 0.1x0.1 squares
if "bc_df" not in st.session_state:
    st.session_state.bc_df = pd.DataFrame(
        [[0.0, 0.0, 0.1, 0.1, "Fixed"], [4.0, 0.0, 0.1, 0.1, "Fixed"]],
        columns=["X (m)", "Y (m)", "Width", "Height", "Type"]
    )

# Point loads every 6 meters on both sides with Fx=2000
if "force_df" not in st.session_state:
    forces = []
    for y in [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]:
        forces.append([0.0, y, 0.1, 0.1, 200.0, 0.0])
        forces.append([4.0, y, 0.1, 0.1, 200.0, 0.0])
        
    st.session_state.force_df = pd.DataFrame(
        forces,
        columns=["X (m)", "Y (m)", "Width", "Height", "Fx (N)", "Fy (N)"]
    )

if "run_bc_df" not in st.session_state:
    st.session_state.run_bc_df = st.session_state.bc_df.copy()
if "run_force_df" not in st.session_state:
    st.session_state.run_force_df = st.session_state.force_df.copy()

if "show_labels" not in st.session_state:
    st.session_state.show_labels = False

# Plotly toolbar config dictionary to ensure standard interactive tools are visible
PLOTLY_CONFIG = {
    'displayModeBar': True,
    'scrollZoom': True
}

# ==========================================
# PART 2: MODEL CONFIGURATION
# ==========================================
st.markdown('<div class="section-header">⚙️ Model Configuration</div>', unsafe_allow_html=True)

# Changed to 4 columns to fit Material Properties here
conf_col1, conf_col2, conf_col3, conf_col4 = st.columns(4)

with conf_col1:
    with st.expander("📏 Domain & Mesh", expanded=False):
        dimx = st.number_input("Domain X (m)", value=4.0, step=1.0, min_value=1.0)
        dimy = st.number_input("Domain Y (m)", value=24.0, step=1.0, min_value=1.0)
        
        mesh_size = st.number_input("Mesh Size (m)", value=0.10, step=0.01, min_value=0.001, format="%.3f")
        
        nelx = max(1, int(round(dimx / mesh_size)))
        nely = max(1, int(round(dimy / mesh_size)))
        
        total_elements = nelx * nely
        if total_elements > 50000:
            st.error(f"🚨 Mesh is too fine! Total elements: {total_elements:,}. The max allowed is 50,000. Please increase the Mesh Size.")
            st.stop()
        else:
            st.success(f"Grid: {nelx} x {nely} elements\n\nTotal: {total_elements:,}")

with conf_col2:
    with st.expander("🧪 Material Properties", expanded=False):
        E = st.number_input("Elastic Modulus (Pa)", value=1500000, step=100000)
        nu = st.slider("Poisson's Ratio (v)", 0.0, 0.5, 0.30)
        rho = st.number_input("Material Density (p)", value=2.400, format="%.3f")
        self_weight = st.checkbox("Include Self-Weight", value=False)


with conf_col3:
    with st.expander("🎯 Optimization Settings", expanded=False):
        vol_frac = st.slider("Volume Fraction", 0.01, 1.0, 0.5)
        rmin = st.number_input("Filter Radius (rmin)", value=0.2, step=0.1)
        itmax = st.number_input("Max Iterations", value=50, step=10)

with conf_col4:
    with st.expander("📐 Thickness Limits", expanded=False):
        tmin = st.number_input("Min Thickness (m)", value=0.1, step=0.10, format="%.4f")
        tmax = st.number_input("Max Thickness (m)", value=1.0, step=0.25, format="%.4f")

st.markdown("---")

# ==========================================
# DYNAMIC COLORMAP GENERATION
# ==========================================
ntmin = np.clip(tmin / (tmax + 1e-9), 0.001, 0.999)

fast_cmap = LinearSegmentedColormap.from_list(
    "custom_copper", 
    [(0.0, '#ffffff'), (ntmin, '#000000'), (1.0, '#ce8d50')]
)

custom_copper_scale = [
    [0.0, '#ffffff'],    
    [ntmin, '#000000'],  
    [1.0, '#ce8d50']     
]

# ==========================================
# PART 3: BOUNDARY CONDITIONS & FORCES
# ==========================================
col_bc, col_run = st.columns(2)

# --- 3A. INTERACTIVE PLOT COLUMN ---
with col_bc:
    st.markdown('<div class="section-header">🎛️ Setup (Click to edit)</div>', unsafe_allow_html=True)
    
    if 'add_bc' not in st.session_state: st.session_state.add_bc = False
    if 'del_bc' not in st.session_state: st.session_state.del_bc = False
    if 'add_fc' not in st.session_state: st.session_state.add_fc = False
    if 'del_fc' not in st.session_state: st.session_state.del_fc = False

    def on_toggle_add_bc():
        if st.session_state.add_bc: st.session_state.update(del_bc=False, add_fc=False, del_fc=False)
    def on_toggle_del_bc():
        if st.session_state.del_bc: st.session_state.update(add_bc=False, add_fc=False, del_fc=False)
    def on_toggle_add_fc():
        if st.session_state.add_fc: st.session_state.update(add_bc=False, del_bc=False, del_fc=False)
    def on_toggle_del_fc():
        if st.session_state.del_fc: st.session_state.update(add_bc=False, del_bc=False, add_fc=False)

    col_t1, col_t2 = st.columns(2)
    col_t1.toggle("➕ ADD Support", key="add_bc", on_change=on_toggle_add_bc)
    col_t2.toggle("➖ DEL Support", key="del_bc", on_change=on_toggle_del_bc)
    col_t3, col_t4 = st.columns(2)
    col_t3.toggle("⚡ ADD Force", key="add_fc", on_change=on_toggle_add_fc)
    col_t4.toggle("❌ DEL Force", key="del_fc", on_change=on_toggle_del_fc)

    fig2d = go.Figure()

    # Base Domain
    fig2d.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                    line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")

    # Plot Supports (Red)
    for i, row in st.session_state.bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        fig2d.add_shape(type="rect", x0=row['X (m)']-hx, y0=row['Y (m)']-hy, x1=row['X (m)']+hx, y1=row['Y (m)']+hy, 
                        line=dict(color='red', width=2), fillcolor='red', opacity=0.6)
        if st.session_state.show_labels:
            fig2d.add_annotation(x=row['X (m)'], y=row['Y (m)'], text=f"S{i+1}", showarrow=False, 
                                 font=dict(color="black", size=11, family="Arial Black"))

    # Plot Forces (Blue)
    for i, row in st.session_state.force_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        fig2d.add_shape(type="rect", x0=row['X (m)']-hx, y0=row['Y (m)']-hy, x1=row['X (m)']+hx, y1=row['Y (m)']+hy, 
                        line=dict(color='blue', width=2), fillcolor='blue', opacity=0.6)
        if st.session_state.show_labels:
            fig2d.add_annotation(x=row['X (m)'], y=row['Y (m)']+1, text=f"F{i+1}", showarrow=False, 
                                 font=dict(color="blue", size=11, family="Arial Black"))

    # Interaction Grid
    grid_spacing = 1.0
    grid_x, grid_y = np.meshgrid(np.arange(0, dimx + 0.1, grid_spacing), np.arange(0, dimy + 0.1, grid_spacing))
    gx, gy = grid_x.flatten(), grid_y.flatten()

    grid_opacity = 0.3 if any([st.session_state.add_bc, st.session_state.del_bc, st.session_state.add_fc, st.session_state.del_fc]) else 0.0
    grid_color = 'green' if (st.session_state.add_bc or st.session_state.add_fc) else 'black'

    fig2d.add_trace(go.Scatter(
        x=gx, y=gy, mode='markers',
        marker=dict(size=12, color=grid_color, opacity=grid_opacity, symbol='square'),
        hoverinfo='text', text="Click here", name="Grid"
    ))

    fig2d.update_layout(
        height=600,       
        autosize=True,
        xaxis=dict(range=[-2, dimx+2], constrain='domain'), 
        yaxis=dict(range=[-2, dimy+2], scaleanchor="x", scaleratio=1, constrain='domain'), 
        clickmode='event+select', 
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False
    )

    # Note the added config parameter here
    event = st.plotly_chart(fig2d, on_select="rerun", key="setup_map", use_container_width=True, config=PLOTLY_CONFIG)

    # Handle Clicks
    if event and "selection" in event and len(event["selection"]["points"]) > 0:
        pt = event["selection"]["points"][0]
        cx, cy = pt['x'], pt['y']
        
        # Add Support
        if st.session_state.add_bc:
            if not ((st.session_state.bc_df['X (m)'] == cx) & (st.session_state.bc_df['Y (m)'] == cy)).any():
                new_row = pd.DataFrame([[float(cx), float(cy), 0.1, 0.1, "Fixed"]], columns=st.session_state.bc_df.columns)
                st.session_state.bc_df = pd.concat([st.session_state.bc_df, new_row], ignore_index=True)
                st.rerun()
                
        # Delete Support
        elif st.session_state.del_bc:
            to_drop = []
            for i, row in st.session_state.bc_df.iterrows():
                if (row['X (m)']-row['Width']/2 <= cx <= row['X (m)']+row['Width']/2) and (row['Y (m)']-row['Height']/2 <= cy <= row['Y (m)']+row['Height']/2):
                    to_drop.append(i)
            if to_drop:
                st.session_state.bc_df = st.session_state.bc_df.drop(to_drop).reset_index(drop=True)
                st.rerun()
                
        # Add Force
        elif st.session_state.add_fc:
            if not ((st.session_state.force_df['X (m)'] == cx) & (st.session_state.force_df['Y (m)'] == cy)).any():
                new_row = pd.DataFrame([[float(cx), float(cy), 0.1, 0.1, 2000.0, 0.0]], columns=st.session_state.force_df.columns)
                st.session_state.force_df = pd.concat([st.session_state.force_df, new_row], ignore_index=True)
                st.rerun()
                
        # Delete Force
        elif st.session_state.del_fc:
            to_drop = []
            for i, row in st.session_state.force_df.iterrows():
                if (row['X (m)']-row['Width']/2 <= cx <= row['X (m)']+row['Width']/2) and (row['Y (m)']-row['Height']/2 <= cy <= row['Y (m)']+row['Height']/2):
                    to_drop.append(i)
            if to_drop:
                st.session_state.force_df = st.session_state.force_df.drop(to_drop).reset_index(drop=True)
                st.rerun()

    # --- UI ORGANIZATION VIA TABS ---
    with st.expander("🛠️ Modify Boundary Conditions & Forces", expanded=False):
        tab_labels, tab_bc, tab_fc, tab_info = st.tabs(["👁️ Labels", "📋 Supports", "⚡ Forces", "ℹ️ Info"])
        
        with tab_labels:
            st.checkbox("🏷️ Show Identifiers on Setup Plot", key="show_labels")
            
        with tab_bc:
            display_df = st.session_state.bc_df.copy()
            display_df.insert(0, "ID", [f"S{i+1}" for i in range(len(display_df))])
            edited_bc_df = st.data_editor(
                display_df, num_rows="dynamic", use_container_width=True, hide_index=True, 
                column_config={"ID": st.column_config.TextColumn(disabled=True), 
                               "Type": st.column_config.SelectboxColumn("Type", options=["Fixed", "Pinned X", "Pinned Y"])}
            )
            if not edited_bc_df.drop(columns=["ID"]).equals(st.session_state.bc_df):
                st.session_state.bc_df = edited_bc_df.drop(columns=["ID"])
                st.rerun()

        with tab_fc:
            display_f_df = st.session_state.force_df.copy()
            display_f_df.insert(0, "ID", [f"F{i+1}" for i in range(len(display_f_df))])
            edited_f_df = st.data_editor(
                display_f_df, num_rows="dynamic", use_container_width=True, hide_index=True,
                column_config={"ID": st.column_config.TextColumn(disabled=True)}
            )
            if not edited_f_df.drop(columns=["ID"]).equals(st.session_state.force_df):
                st.session_state.force_df = edited_f_df.drop(columns=["ID"])
                st.rerun()
                
        with tab_info:
            st.markdown("""
            **Supports:**
            * **Fixed:** Prevents translation in both X and Y directions.
            * **Pinned X:** Prevents translation in the X direction only (acts as a roller).
            * **Pinned Y:** Prevents translation in the Y direction only (acts as a roller).
            
            **Forces:**
            * Specify the external point loads in the X and Y directions (`Fx` and `Fy`). Downward/Leftward forces should be negative.
            """)

# --- 3B. SOLVER / RUN COLUMN ---
with col_run:
    st.markdown('<div class="section-header">🚀 Solver</div>', unsafe_allow_html=True)
    
    solver_bc_df = st.session_state.bc_df.copy()
    solver_bc_df["Type"] = solver_bc_df["Type"].map({"Fixed": 1, "Pinned X": 2, "Pinned Y": 3})
    BCMatrix = solver_bc_df.to_numpy()
    
    ForceMatrix = st.session_state.force_df.to_numpy()

    run_pressed = st.button("🚀 Run Optimization", type="primary", use_container_width=True)
    
    # Placeholders for live plotting and status
    color_bar_spot = st.empty()
    live_plot_spot = st.empty()
    status_text = st.empty()

    def plot_2d_live_fast(Z_matrix):
        Z_norm = Z_matrix / (tmax + 1e-9)
        Z_norm = np.clip(Z_norm, 0, 1)
        rgba_img = fast_cmap(Z_norm)
        return rgba_img

    def plot_2d_thickness_plotly(Z_matrix):
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=np.flipud(Z_matrix),
            x=np.linspace(0, dimx, Z_matrix.shape[1]),
            y=np.linspace(0, dimy, Z_matrix.shape[0]),
            colorscale=custom_copper_scale,
            zmin=0, zmax=tmax, 
            showscale=True, 
            colorbar=dict(
                title='Element Thickness (m)', 
                orientation='h', x=0.5, y=1.05, 
                xanchor='center', yanchor='bottom', 
                thickness=15, len=0.8
            ), 
            hoverinfo='skip'
        ))
        
        fig.add_shape(type="rect", x0=0, y0=0, x1=dimx, y1=dimy, 
                      line=dict(color="#0f172a", width=2, dash="dash"), fillcolor="rgba(0,0,0,0)")
        
        for i, row in st.session_state.run_bc_df.iterrows():
            hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
            fig.add_shape(type="rect", x0=row['X (m)']-hx, y0=row['Y (m)']-hy, x1=row['X (m)']+hx, y1=row['Y (m)']+hy, 
                          line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.5)')
                          
        for i, row in st.session_state.run_force_df.iterrows():
            hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
            fig.add_shape(type="rect", x0=row['X (m)']-hx, y0=row['Y (m)']-hy, x1=row['X (m)']+hx, y1=row['Y (m)']+hy, 
                          line=dict(color='blue', width=1), fillcolor='rgba(0,0,255,0.5)')
            
        fig.update_layout(
            height=600,
            autosize=True,
            xaxis=dict(range=[-2, dimx+2], constrain='domain'), 
            yaxis=dict(range=[-2, dimy+2], scaleanchor="x", scaleratio=1, constrain='domain'), 
            margin=dict(l=0, r=0, t=0, b=0), showlegend=False
        )
        return fig

    if run_pressed:
        if len(BCMatrix) == 0: 
            st.error("Please add at least one support!")
        elif len(ForceMatrix) == 0 and not self_weight: 
            st.error("Add a force or enable self-weight!")
        else:
            st.session_state.run_bc_df = st.session_state.bc_df.copy()
            st.session_state.run_force_df = st.session_state.force_df.copy()
            
            target_volume = (vol_frac * dimx * dimy * tmin)
            
            gradient_html = f"""
            <div style="text-align: center; margin-bottom: 5px; font-weight: bold; color: #475569; font-size: 0.9rem;">Element Thickness (m)</div>
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; padding: 0 10px;">
                <span style="font-size: 0.8rem; font-weight: bold;">0</span>
                <div style="flex-grow: 1; height: 12px; margin: 0 10px; background: linear-gradient(to right, #ffffff 0%, #000000 {ntmin*100}%, #b87333 100%); border: 1px solid #cbd5e1; border-radius: 4px;"></div>
                <span style="font-size: 0.8rem; font-weight: bold;">{tmax}</span>
            </div>
            """
            color_bar_spot.markdown(gradient_html, unsafe_allow_html=True)

            def update_live_view(current_it, current_ch, current_Z):
                live_plot_spot.image(plot_2d_live_fast(current_Z), use_container_width=True) 
                status_text.info(f"⚙️ Optimizing... Iteration: {current_it}")

            with st.spinner("Optimizing..."):
                SW_val = 1 if self_weight else 0
                X, Y, Thickness, history = logic.run_topology_optimization(
                    float(dimx), float(dimy), float(E), float(nu), float(rho), int(SW_val), 
                    BCMatrix, ForceMatrix, int(nelx), int(nely), float(target_volume), 
                    float(rmin), float(tmin), float(tmax), int(itmax), progress_callback=update_live_view
                )
                st.session_state.history, st.session_state.X, st.session_state.Y, st.session_state.run_finished = history, X, Y, True
                st.rerun()

    if st.session_state.run_finished and st.session_state.history is not None:
        color_bar_spot.empty() 
        final_plotly_fig = plot_2d_thickness_plotly(st.session_state.history[-1])
        # Added config parameter here
        live_plot_spot.plotly_chart(final_plotly_fig, use_container_width=True, key="final_result_plot", config=PLOTLY_CONFIG)
        status_text.success(f"✅ Optimization Complete! Iterations run: {len(st.session_state.history)}")

# ==========================================
# PART 4: INTERACTIVE 3D RESULTS (SOLID VOLUMETRIC EXTRUSION)
# ==========================================
if st.session_state.run_finished:
    st.markdown("---")
    st.markdown('<div class="section-header">🕒 Interactive 3D Results</div>', unsafe_allow_html=True)
    
    with st.expander("🖱️ How to interact with the 3D Plot", expanded=False):
        st.markdown("""
        **On a Computer (Mouse):**
        * **Rotate:** Left-click and drag.
        * **Pan:** Right-click and drag (or `Shift` + Left-click).
        * **Zoom:** Use the mouse scroll wheel.
        """)

    steps = len(st.session_state.history)
    plot_placeholder = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    idx = st.slider("Iteration History", 0, steps - 1, steps - 1)
    
    col_cam, col_scale = st.columns(2)
    with col_cam:
        view_choice = st.selectbox("🎥 Camera View", ["Default", "Top", "Front", "Side"])
    with col_scale:
        use_true_scale = st.checkbox("📏 True Thickness Scale", value=True)
        if use_true_scale:
            z_scale_pct = int(100*tmax/max(dimx, dimy))
        else:
            if "z_scale_val" not in st.session_state: st.session_state.z_scale_val = int(100*tmax/max(dimx, dimy))
            z_scale_pct = st.slider("Visual Thickness Scale (%)", int(100*tmax/max(dimx, dimy)), 100, st.session_state.z_scale_val)
            st.session_state.z_scale_val = z_scale_pct

    if view_choice == "Front": cam_eye, cam_up = dict(x=0, y=2.5, z=0), dict(x=0, y=0, z=1)
    elif view_choice == "Top": cam_eye, cam_up = dict(x=0, y=0, z=2.5), dict(x=0, y=1, z=0)
    elif view_choice == "Side": cam_eye, cam_up = dict(x=2.5, y=0, z=0), dict(x=0, y=1, z=0)
    else: cam_eye, cam_up = dict(x=1.8, y=1.8, z=1.8), dict(x=0, y=0, z=1) 
    
    Z_raw = st.session_state.history[idx]
    Z_final = np.flipud(Z_raw) 
    
    @st.cache_data(show_spinner=False)
    def generate_3d_mesh(Z_matrix, tmin_val, tmax_val, dimx_val, dimy_val):
        void_threshold = 0.1 * tmin_val 
        Z_filtered = np.where(Z_matrix > void_threshold, Z_matrix, 0.0)
        Z_padded = np.pad(Z_filtered, pad_width=1, mode='constant', constant_values=0)
        
        nz = 7 
        z_vals = np.linspace(-tmax_val/2 * 1.1, tmax_val/2 * 1.1, nz)
        ny_pad, nx_pad = Z_padded.shape
        
        V = np.zeros((ny_pad, nx_pad, nz))
        for k, z in enumerate(z_vals):
            V[:, :, k] = (Z_padded / 2.0) - np.abs(z) - 1e-4
            
        verts, faces, _, _ = measure.marching_cubes(V, level=0.0)
        
        y_idx = verts[:, 0] - 1 
        x_idx = verts[:, 1] - 1
        z_idx = verts[:, 2]
        
        dx_val_phys = dimx_val / (Z_matrix.shape[1] - 1)
        dy_val_phys = dimy_val / (Z_matrix.shape[0] - 1)
        dz_val_phys = (tmax_val * 1.1) / (nz - 1) 
        
        x_phys = x_idx * dx_val_phys
        y_phys = y_idx * dy_val_phys
        z_phys = z_vals[0] + z_idx * dz_val_phys
        
        v_y_idx = np.clip(verts[:, 0].astype(int), 0, ny_pad-1)
        v_x_idx = np.clip(verts[:, 1].astype(int), 0, nx_pad-1)
        vertex_thickness = Z_padded[v_y_idx, v_x_idx]
        
        return x_phys, y_phys, z_phys, faces, vertex_thickness

    x_p, y_p, z_p, f_idx, v_thick = generate_3d_mesh(Z_final, tmin, tmax, dimx, dimy)

    solid_mesh = go.Mesh3d(
        x=x_p, 
        y=z_p, 
        z=y_p,
        i=f_idx[:, 0], j=f_idx[:, 1], k=f_idx[:, 2],
        intensity=v_thick,
        colorscale=custom_copper_scale, 
        cmin=0, cmax=tmax,
        showscale=True,
        colorbar=dict(title='Element Thickness (m)', orientation='h', x=0.5, y=1.05, xanchor='center', yanchor='bottom', thickness=15, len=0.6),
        opacity=1.0,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.9, specular=0.1, roughness=0.5) 
    )

    fig = go.Figure(data=[solid_mesh])

    support_min_t, support_max_t = -tmax * 0.6, tmax * 0.6
    for i, row in st.session_state.run_bc_df.iterrows():
        hx, hy = row['Width'] / 2.0, row['Height'] / 2.0
        x_min, x_max = row['X (m)'] - hx, row['X (m)'] + hx
        y_min, y_max = row['Y (m)'] - hy, row['Y (m)'] + hy
        
        fig.add_trace(go.Mesh3d(
            x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
            y=[support_min_t, support_min_t, support_min_t, support_min_t, support_max_t, support_max_t, support_max_t, support_max_t],
            z=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            color='darkred', opacity=0.8, flatshading=True, showlegend=False
        ))

    max_dim = max(dimx, dimy)
    aspect_x = dimx / max_dim
    aspect_z = dimy / max_dim
    
    if use_true_scale:
        true_thick_ratio = max(0.015, (tmax * 1.1) / max_dim)
        scene_layout = dict(
            xaxis=dict(title='X (m)'),
            yaxis=dict(title='Thickness (m)'),
            zaxis=dict(title='Y (m)'),
            aspectmode='manual',
            aspectratio=dict(x=aspect_x, y=true_thick_ratio, z=aspect_z),
            camera=dict(eye=cam_eye, up=cam_up)
        )
    else:
        scene_layout = dict(
            xaxis=dict(range=[-0.05 * dimx, 1.05 * dimx], title='X (m)'),
            yaxis=dict(range=[-tmax, tmax], title='Thickness (m)'),
            zaxis=dict(range=[-0.05 * dimy, 1.05 * dimy], title='Y (m)'),
            aspectmode='manual', 
            aspectratio=dict(x=aspect_x, y=z_scale_pct/100.0, z=aspect_z),
            camera=dict(eye=cam_eye, up=cam_up)
        )

    fig.update_layout(
        scene=scene_layout,
        margin=dict(l=0, r=0, b=0, t=50), height=600 
    )
    
    # Added config parameter here
    plot_placeholder.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # STL Export
    st.markdown("---")
    st.subheader("💾 Export Solid Geometry")
    
    final_x_p, final_y_p, final_z_p, final_f_idx, _ = generate_3d_mesh(np.flipud(st.session_state.history[-1]), tmin, tmax, dimx, dimy)
    
    def generate_stl_3d(vx, vy, vz, f_indices):
        solid_mesh_stl = mesh.Mesh(np.zeros(f_indices.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(f_indices):
            for j in range(3):
                solid_mesh_stl.vectors[i][j] = [vx[f[j]], vz[f[j]], vy[f[j]]]
        buf = io.BytesIO()
        solid_mesh_stl.save('membrane.stl', fh=buf)
        return buf.getvalue()

    stl_data = generate_stl_3d(final_x_p, final_y_p, final_z_p, final_f_idx)
    st.download_button("📥 Download Final Solid Model (.STL)", data=stl_data, file_name=f"Optimized_Solid_Final.stl", mime="model/stl", type="primary")

# ==========================================
# PART 5: AUTHOR & CONTACT INFO
# ==========================================
st.markdown("---")
st.markdown('<div class="section-header">📬 Contact & Info</div>', unsafe_allow_html=True)

col_info1, col_info2 = st.columns([2, 1])
with col_info1:
    st.markdown("""
    **Created by:** Sebastian Pozo Ocampo  
    **Contact:** [sebaspozo94@gmail.com](mailto:sebaspozo94@gmail.com)
    *For custom workflows, project-specific studies, or collaboration.*
    """)
with col_info2:
    st.markdown("""
    *Connect with me:*
    * [Website](https://streamline-gallery-5d621e11.buildaispace.app)  
    * [LinkedIn](https://www.linkedin.com/in/sebastianpozo94/)
    * [GitHub](https://github.com/sebaspozo94)
    """)
