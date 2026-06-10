import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Spin Coating Simulator",
    layout="wide"
)

st.title("Spin Coating Thin-Film Simulator")

st.markdown("""
This simulator is based on the Emslie–Bonner–Peck model with solvent evaporation
and time-dependent viscosity growth. Users can explore how spin-coating parameters
affect film thickness, radial uniformity, and edge-bead behavior.
""")

# Sidebar Inputs
st.sidebar.header("Process Inputs")
st.sidebar.caption(
    "Adjust spin-coating parameters and observe their effects on thickness evolution, "
    "validation results, and coating uniformity."
)

st.sidebar.markdown("### Rotational Speed (RPM)")
rpm_num = st.sidebar.number_input(
    "Enter RPM",
    min_value=500,
    max_value=6000,
    value=3000,
    step=100
)
rpm = st.sidebar.slider(
    "Adjust RPM",
    min_value=500,
    max_value=6000,
    value=rpm_num,
    step=5
)

st.sidebar.markdown("### Initial Viscosity (Pa·s)")
eta0_num = st.sidebar.number_input(
    "Enter Initial Viscosity",
    min_value=0.005,
    max_value=0.5,
    value=0.05,
    step=0.005,
    format="%.3f"
)
eta0 = st.sidebar.slider(
    "Adjust Initial Viscosity",
    min_value=0.005,
    max_value=0.5,
    value=eta0_num,
    step=0.005,
    format="%.3f"
)

st.sidebar.markdown("### Initial Thickness (μm)")
h0_num = st.sidebar.number_input(
    "Enter Initial Thickness",
    min_value=10,
    max_value=300,
    value=100,
    step=10
)
h0_um = st.sidebar.slider(
    "Adjust Initial Thickness",
    min_value=10,
    max_value=300,
    value=h0_num,
    step=1
)

st.sidebar.markdown("### Evaporation Rate (μm/s)")
E_num = st.sidebar.number_input(
    "Enter Evaporation Rate",
    min_value=0.001,
    max_value=1.0,
    value=0.05,
    step=0.001,
    format="%.3f"
)
E_um_s = st.sidebar.slider(
    "Adjust Evaporation Rate",
    min_value=0.001,
    max_value=1.0,
    value=E_num,
    step=0.001,
    format="%.3f"
)

# Constants
rho = 1000
beta = 0.15
total_time = 60
dt = 0.01
wafer_radius_mm = 75
gel_viscosity = 5.0

# Unit Conversion
omega = rpm * 2 * np.pi / 60
h0 = h0_um * 1e-6
E = E_um_s * 1e-6

# Main Simulation: evaporation + viscosity growth
time_list = []
h_list = []
eta_list = []

t = 0
h = h0

while t <= total_time and h > 0:
    time_list.append(t)
    h_list.append(h * 1e6)

    eta = eta0 * np.exp(beta * t)
    eta_list.append(eta)

    dhdt = -(2 * rho * omega**2 / (3 * eta)) * h**3 - E
    h += dhdt * dt

    if h < 0:
        h = 0

    t += dt

time_arr = np.array(time_list)
h_num = np.array(h_list)
eta_arr = np.array(eta_list)

# Validation Simulation: beta = 0, E = 0
h_val = h0
h_val_list = []

for _ in time_arr:
    h_val_list.append(h_val * 1e6)

    dhdt_val = -(2 * rho * omega**2 / (3 * eta0)) * h_val**3
    h_val += dhdt_val * dt

    if h_val < 0:
        h_val = 0

h_validation = np.array(h_val_list)

# Analytical EBP Solution: constant viscosity, no evaporation
K = (4 * rho * omega**2 * h0**2) / (3 * eta0)
h_ana = (h0 / np.sqrt(1 + K * time_arr)) * 1e6

# Final Thickness
final_thickness = h_num[-1]

# Gel Time Prediction
if gel_viscosity > eta0:
    t_gel = np.log(gel_viscosity / eta0) / beta
else:
    t_gel = 0

# Edge Bead Model
r = np.linspace(0, wafer_radius_mm, 200)

# Phenomenological edge-bead strength:
# Higher rpm improves radial spreading, while higher viscosity increases edge accumulation.
rpm_factor = 3000 / rpm
viscosity_factor = eta0 / 0.05

edge_strength = 0.12 * rpm_factor * viscosity_factor
edge_strength = np.clip(edge_strength, 0.02, 0.40)

edge_width = 0.12 * wafer_radius_mm

edge_bead = final_thickness * edge_strength * np.exp(
    -((wafer_radius_mm - r) / edge_width) ** 2
)

radial_thickness = final_thickness + edge_bead

avg_h = np.mean(radial_thickness)
max_h = np.max(radial_thickness)
min_h = np.min(radial_thickness)

uniformity_error = ((max_h - min_h) / avg_h) * 100
target_uniformity = 2.0

# Metrics
st.subheader("Simulation Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Final Thickness", f"{final_thickness:.2f} μm")
col2.metric("Average Thickness", f"{avg_h:.2f} μm")
col3.metric("Max Thickness", f"{max_h:.2f} μm")
col4.metric("Uniformity Error", f"{uniformity_error:.2f}%")
col5.metric("Predicted t_gel", f"{t_gel:.1f} s")

if uniformity_error <= target_uniformity:
    st.success(
        f"✅ Uniformity target satisfied: {uniformity_error:.2f}% ≤ ±{target_uniformity:.1f}%"
    )
else:
    st.error(
        f"❌ Uniformity target not satisfied: {uniformity_error:.2f}% > ±{target_uniformity:.1f}%"
    )

st.caption(
    "Uniformity error represents radial thickness variation across the wafer. "
    "A lower value indicates a more uniform photoresist coating."
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "Core Simulation",
    "Validation",
    "Challenge Mode"
])

with tab1:
    st.subheader("Film Thickness Evolution")

    fig1, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(
        time_arr,
        h_num,
        label="Numerical Simulation: evaporation + viscosity growth",
        linewidth=2
    )

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Film Thickness (μm)")
    ax1.set_title("Film Thickness Evolution")
    ax1.grid(True)
    ax1.legend()

    st.pyplot(fig1)

    st.subheader("Radial Thickness Distribution")

    fig3, ax3 = plt.subplots(figsize=(8, 4))

    ax3.plot(
        r,
        radial_thickness,
        linewidth=2
    )

    ax3.set_xlabel("Radial Position (mm)")
    ax3.set_ylabel("Film Thickness (μm)")
    ax3.set_title("Radial Thickness Distribution with Simplified Edge Bead Model")
    ax3.grid(True)

    st.pyplot(fig3)

    st.caption(
        "Note: The edge-bead profile is a simplified qualitative model used for visualization, "
        "not a full hydrodynamic edge-bead simulation."
    )

with tab2:
    st.subheader("Validation View: Numerical EBP vs Analytical EBP")

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        time_arr,
        h_validation,
        label="Numerical EBP: β = 0, E = 0",
        linewidth=2
    )

    ax2.plot(
        time_arr,
        h_ana,
        "--",
        label="Analytical EBP Solution",
        linewidth=2
    )

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Film Thickness (μm)")
    ax2.set_title("Validation of Numerical Solver")
    ax2.grid(True)
    ax2.legend()

    st.pyplot(fig2)

    st.subheader("Validation Results")

    check_times = [1, 5, 10, 30, 60]
    table_data = []

    for ct in check_times:
        idx = np.argmin(np.abs(time_arr - ct))

        num_val = h_validation[idx]
        ana_val = h_ana[idx]

        error = abs(num_val - ana_val) / ana_val * 100

        table_data.append(
            {
                "Time (s)": ct,
                "Numerical EBP (μm)": round(num_val, 3),
                "Analytical EBP (μm)": round(ana_val, 3),
                "Error (%)": round(error, 2)
            }
        )

    st.table(table_data)

    st.info(
        "Validation is performed under the analytical EBP limit: constant viscosity and zero evaporation. "
        "Small errors mainly come from the Forward Euler time discretization."
    )

with tab3:
    st.subheader("Challenge Mode: Find Conditions for ±2% Uniformity")

    st.write(
        "This mode evaluates whether the selected process condition satisfies the target "
        "coating-uniformity specification. Users can adjust rotational speed and viscosity "
        "in the sidebar to explore suitable operating conditions."
    )

    if uniformity_error <= target_uniformity:
        st.success(
            f"✅ Passed: This condition satisfies the ±2% uniformity specification. "
            f"Uniformity error = {uniformity_error:.2f}%"
        )
    else:
        st.warning(
            f"⚠️ Failed: This condition does not satisfy the ±2% uniformity specification. "
            f"Uniformity error = {uniformity_error:.2f}%"
        )

    st.markdown("""
    **Design interpretation**

    - Higher rotational speed generally reduces final film thickness and improves radial spreading.
    - Lower viscosity promotes stronger radial flow and can improve uniformity.
    - Excessively high viscosity suppresses radial spreading and may increase thickness variation.
    - Solvent evaporation and viscosity growth reduce radial flow over time.
    - Edge-bead formation can increase radial non-uniformity near the wafer boundary.
    """)

st.success("Simulation completed successfully.")
