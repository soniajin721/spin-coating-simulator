import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Spin Coating Simulator",
    layout="wide"
)

st.title("Spin Coating Thin-Film Simulator")

st.markdown("""
This simulator is based on the Emslie–Bonner–Peck model with
solvent evaporation and time-dependent viscosity growth.
""")

# Sidebar Inputs
st.sidebar.header("Process Parameters")

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    500,
    6000,
    3000
)

eta0 = st.sidebar.slider(
    "Initial Viscosity (Pa·s)",
    0.005,
    0.5,
    0.05
)

h0_um = st.sidebar.slider(
    "Initial Thickness (μm)",
    10,
    300,
    100
)

E_um_s = st.sidebar.slider(
    "Evaporation Rate (μm/s)",
    0.001,
    1.0,
    0.05
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
    eta = eta0 * np.exp(beta * t)
    dhdt = -(2 * rho * omega**2 / (3 * eta)) * h**3 - E

    h += dhdt * dt

    if h < 0:
        h = 0

    time_list.append(t)
    h_list.append(h * 1e6)
    eta_list.append(eta)

    t += dt

time_arr = np.array(time_list)
h_num = np.array(h_list)
eta_arr = np.array(eta_list)

# Validation Simulation: beta = 0, E = 0
# This numerical solution uses the same physical assumptions as the analytical EBP solution.
h_val = h0
h_val_list = []

for _ in time_arr:
    dhdt_val = -(2 * rho * omega**2 / (3 * eta0)) * h_val**3
    h_val += dhdt_val * dt

    if h_val < 0:
        h_val = 0

    h_val_list.append(h_val * 1e6)

h_validation = np.array(h_val_list)

# Analytical EBP Solution: constant viscosity, no evaporation
K = (4 * rho * omega**2 * h0**2) / (3 * eta0)

h_ana = (
    h0 / np.sqrt(1 + K * time_arr)
) * 1e6

# Final Thickness
final_thickness = h_num[-1]

# Gel Time Prediction
if gel_viscosity > eta0:
    t_gel = np.log(gel_viscosity / eta0) / beta
else:
    t_gel = 0

# Edge Bead Model
r = np.linspace(0, wafer_radius_mm, 200)

edge_strength = 0.12
edge_width = 0.12 * wafer_radius_mm

edge_bead = final_thickness * edge_strength * np.exp(
    -((wafer_radius_mm - r) / edge_width) ** 2
)

radial_thickness = final_thickness + edge_bead

avg_h = np.mean(radial_thickness)
max_h = np.max(radial_thickness)
min_h = np.min(radial_thickness)

uniformity_error = ((max_h - min_h) / avg_h) * 100

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Final Thickness",
    f"{final_thickness:.2f} μm"
)

col2.metric(
    "Average Thickness",
    f"{avg_h:.2f} μm"
)

col3.metric(
    "Max Thickness",
    f"{max_h:.2f} μm"
)

col4.metric(
    "Uniformity Error",
    f"{uniformity_error:.2f}%"
)

col5.metric(
    "Predicted t_gel",
    f"{t_gel:.1f} s"
)

# Main Simulation Plot
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

# Validation Plot
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

# Radial Thickness Plot
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

# Validation Table
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

# Challenge Mode
st.subheader("Challenge Mode: ±2% Uniformity Specification")

if uniformity_error <= 2:
    st.success(
        f"This condition satisfies the ±2% uniformity specification. "
        f"Uniformity error = {uniformity_error:.2f}%"
    )
else:
    st.warning(
        f"This condition does not satisfy the ±2% uniformity specification. "
        f"Uniformity error = {uniformity_error:.2f}%"
    )

st.markdown("""
**Design interpretation**

- Higher rotational speed generally reduces final film thickness.
- Lower viscosity promotes stronger radial spreading.
- Solvent evaporation and viscosity growth reduce radial flow over time.
- Edge-bead formation can increase radial non-uniformity near the wafer boundary.
""")

st.success("Simulation completed successfully.")
