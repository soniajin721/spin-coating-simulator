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
solvent evaporation and viscosity growth.
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

# Simulation
omega = rpm * 2 * np.pi / 60
h = h0_um * 1e-6
E = E_um_s * 1e-6

time_list = []
h_list = []

t = 0

while t <= total_time and h > 0:

    eta = eta0 * np.exp(beta * t)

    dhdt = -(2 * rho * omega**2 / (3 * eta)) * h**3 - E

    h += dhdt * dt

    if h < 0:
        h = 0

    time_list.append(t)
    h_list.append(h * 1e6)

    t += dt

time_arr = np.array(time_list)
h_num = np.array(h_list)

# Analytical Solution
K = (4 * rho * omega**2 * (h0_um * 1e-6)**2) / (3 * eta0)

h_ana = (
    (h0_um * 1e-6)
    / np.sqrt(1 + K * time_arr)
) * 1e6

# Final Thickness
final_thickness = h_num[-1]

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

uniformity_error = (
    (max_h - min_h) / avg_h
) * 100

# Metrics
col1, col2, col3, col4 = st.columns(4)

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

# Plot 1
fig1, ax1 = plt.subplots(figsize=(8,4))

ax1.plot(
    time_arr,
    h_num,
    label="Numerical Solution",
    linewidth=2
)

ax1.plot(
    time_arr,
    h_ana,
    "--",
    label="Analytical Solution",
    linewidth=2
)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Film Thickness (μm)")
ax1.set_title("Numerical vs Analytical Solution")
ax1.grid(True)
ax1.legend()

st.pyplot(fig1)

# Plot 2
fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.plot(
    r,
    radial_thickness,
    linewidth=2
)

ax2.set_xlabel("Radial Position (mm)")
ax2.set_ylabel("Film Thickness (μm)")
ax2.set_title("Radial Thickness Distribution with Edge Bead")
ax2.grid(True)

st.pyplot(fig2)

# Validation Table
st.subheader("Validation Results")

check_times = [1, 5, 10, 30, 60]

table_data = []

for ct in check_times:

    idx = np.argmin(np.abs(time_arr - ct))

    num_val = h_num[idx]
    ana_val = h_ana[idx]

    error = abs(num_val - ana_val) / ana_val * 100

    table_data.append(
        {
            "Time (s)": ct,
            "Numerical (μm)": round(num_val, 3),
            "Analytical (μm)": round(ana_val, 3),
            "Error (%)": round(error, 2)
        }
    )

st.table(table_data)

st.success(
    "Simulation completed successfully."
)
