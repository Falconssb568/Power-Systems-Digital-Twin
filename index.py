"""
X-01N POSEIDON-KILO :: Reactor-Coupled Power System Digital Twin
=================================================================
Reduced-order, non-proliferation-relevant ENERGY SYSTEMS model.

This module does NOT model neutronics, criticality, fuel enrichment,
or any weapons-relevant physics. It treats the reactor purely as a
controllable, thermally-lagged heat source (a standard simplification
used in nuclear *energy systems* courses, e.g. MIT 22.312/22.313) and
focuses on:
    - core/coolant thermal-hydraulics (lumped-capacitance)
    - closed Brayton power-conversion cycle thermodynamics
    - coupling of electrical output to a time-varying vehicle load
    - a small mass-optimization study (reactor rating vs. buffer size)

All numerical constants are representative textbook / open-literature
magnitudes (Todreas & Kazimi; El-Genk & Mason, Kilopower open papers),
not a design for any real system.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# -----------------------------------------------------------------------
# 1. MISSION POWER-DEMAND PROFILE  (grounded in the X-01 manuscript)
# -----------------------------------------------------------------------
# X-01 defines total resistance D_total = D_a + D_h + D_int + D_tr and
# instantaneous propulsive power P(t) = D_total(t) * V(t). We do not
# re-derive D_total here; instead we *use* its documented behavior:
# power rises sharply during air<->water transition events (D_int, D_tr
# terms) and settles to lower cruise levels in pure-air or pure-subsea
# flight. That qualitative duty cycle is what the power system must serve.

dt = 1.0                      # s
T_end = 3600.0 * 2             # 2-hour representative mission segment
t = np.arange(0, T_end, dt)

def vehicle_power_demand(t):
    """Representative P(t) [kW] for an X-01-class vehicle:
    cruise baseline + periodic air/water transition spikes (D_tr, D_int
    terms dominate briefly) + sensor/control housekeeping load."""
    cruise = 8.0 + 1.5 * np.sin(2 * np.pi * t / 900.0)         # slow cruise variation
    housekeeping = 0.6                                          # avionics/control, kW
    # Transition events every ~20 minutes, ~90 s duration, sharp power spike
    transition = np.zeros_like(t)
    for t0 in np.arange(300, T_end, 1200):
        mask = (t >= t0) & (t < t0 + 90)
        # smooth spike shape (raised cosine) representing D_int + D_tr surge
        local = (t[mask] - t0) / 90.0
        transition[mask] += 14.0 * np.sin(np.pi * local) ** 2
    noise = rng.normal(0, 0.15, size=t.shape)
    return np.clip(cruise + housekeeping + transition + noise, 0.5, None)

P_demand = vehicle_power_demand(t)   # kW electrical

# -----------------------------------------------------------------------
# 2. REACTOR THERMAL MODEL  (reduced-order, lumped capacitance)
# -----------------------------------------------------------------------
# Core power is a *commanded* quantity (control-drum / rod position sets
# a target thermal power), reached through a first-order thermal lag -
# standard reduced-order treatment for reactor-coupled energy system
# studies at this level (cf. point-kinetics-free "black box" core models).

P_th_rating = 45.0        # kWt, nameplate thermal power of the modeled unit
tau_core = 25.0            # s, core thermal response time constant
C_core = 4.0e4              # J/K, lumped core+coolant thermal mass
m_dot = 0.015               # kg/s, primary coolant mass flow (sized for kWt-class unit)
cp_coolant = 5190.0          # J/kg-K  (He-Xe gas mixture, Brayton-compatible)
T_in = 550.0                # K, coolant inlet (post heat-exchanger) temperature
UA_loss = 4.0                # W/K, parasitic thermal loss to structure/environment
T_ambient = 300.0            # K

T_max_core = 1100.0          # K, material/fuel temperature ceiling (design limit)

# -----------------------------------------------------------------------
# 3. CLOSED BRAYTON POWER-CONVERSION CYCLE
# -----------------------------------------------------------------------
# Standard recuperated closed-Brayton cycle, carried in absolute
# temperatures (K) so every term has clear physical meaning:
#   1 -> 2   compression (real, eta_compressor)
#   2 -> 5   recuperator preheat
#   5 -> 3   heat addition from reactor coolant (T3 ~= core/HX hot side)
#   3 -> 4   expansion through turbine (real, eta_turbine)
#   4 -> 2   recuperator hot side, then external heat rejection to T1
gamma = 5.0 / 3.0            # He-Xe monatomic mixture, k = cp/cv
pressure_ratio = 2.2
eta_turbine = 0.85
eta_compressor = 0.82
eta_recuperator = 0.90
eta_generator = 0.95
T1_compressor_inlet = 320.0   # K, post heat-rejection / radiator temperature
rp_term = pressure_ratio ** ((gamma - 1) / gamma)

def brayton_efficiency(T_hot_K):
    """Recuperated closed-Brayton thermal efficiency given turbine-inlet
    (core/heat-exchanger hot-side) temperature T_hot_K. Returns eta in [0, ~0.45]."""
    T1 = T1_compressor_inlet
    T2s = T1 * rp_term
    T2 = T1 + (T2s - T1) / eta_compressor
    T3 = T_hot_K
    T4s = T3 / rp_term
    T4 = T3 - eta_turbine * (T3 - T4s)
    T5 = T2 + eta_recuperator * (T4 - T2)          # recuperated compressor-exit temp
    q_in = max(T3 - T5, 1e-3)
    w_turbine = T3 - T4
    w_compressor = T2 - T1
    eta = (w_turbine - w_compressor) / q_in
    return float(np.clip(eta, 0.0, 0.45))

# -----------------------------------------------------------------------
# 4. ENERGY-BUFFER (battery) MODEL
# -----------------------------------------------------------------------
E_buffer_capacity = 6.0    # kWh, energy storage buffer sized to smooth transients
SOC = 0.7                   # initial state of charge (fraction)
SOC_min, SOC_max = 0.15, 0.98
eta_battery_rt = 0.93        # round-trip efficiency

# -----------------------------------------------------------------------
# 5. SIMPLE SUPERVISORY CONTROL LAW
# -----------------------------------------------------------------------
# The reactor cannot follow fast transients (tau_core), so it targets a
# *slowly varying* thermal power to hold the buffer near a setpoint,
# while the buffer absorbs/supplies the fast transition spikes.

SOC_setpoint = 0.65
Kp_soc = 60.0   # kWt commanded per unit SOC error (proportional control)

# -----------------------------------------------------------------------
# 6. TIME-MARCHING SIMULATION  (explicit Euler, dt = 1 s)
# -----------------------------------------------------------------------
n = len(t)
P_th = np.zeros(n)
P_cmd = np.zeros(n)
T_core = np.zeros(n)
T_out = np.zeros(n)
eta_cycle = np.zeros(n)
P_elec = np.zeros(n)
SOC_hist = np.zeros(n)
P_th[0] = 20.0
T_core[0] = 900.0
SOC_hist[0] = SOC

for i in range(1, n):
    # --- supervisory command (bounded to nameplate rating) ---
    soc_err = SOC_setpoint - SOC_hist[i - 1]
    P_cmd[i] = np.clip(15.0 + Kp_soc * soc_err, 3.0, P_th_rating)

    # --- core thermal lag ---
    dP = (P_cmd[i] - P_th[i - 1]) / tau_core
    P_th[i] = P_th[i - 1] + dP * dt

    # --- lumped core energy balance ---
    Q_removed = m_dot * cp_coolant * (T_core[i - 1] - T_in) / 1000.0  # kW
    Q_loss = UA_loss * (T_core[i - 1] - T_ambient) / 1000.0            # kW
    dT_core = (P_th[i] - Q_removed - Q_loss) * 1000.0 / C_core
    T_core[i] = T_core[i - 1] + dT_core * dt
    T_core[i] = min(T_core[i], T_max_core * 1.05)  # soft numerical guard

    T_out[i] = T_in + (Q_removed * 1000.0) / (m_dot * cp_coolant)

    # --- power conversion ---
    eta_cycle[i] = brayton_efficiency(T_core[i])
    P_elec[i] = eta_cycle[i] * P_th[i] * eta_generator

    # --- buffer energy balance ---
    net_kW = P_elec[i] - P_demand[i]
    if net_kW < 0:
        net_kWh_step = net_kW * dt / 3600.0            # discharging, no extra loss term applied here
    else:
        net_kWh_step = net_kW * dt / 3600.0 * eta_battery_rt  # charging loss
    SOC_hist[i] = np.clip(SOC_hist[i - 1] + net_kWh_step / E_buffer_capacity, 0.0, 1.0)

# -----------------------------------------------------------------------
# 7. DIAGNOSTICS
# -----------------------------------------------------------------------
print(f"Peak core temperature:      {T_core.max():.1f} K  (limit {T_max_core:.0f} K)")
print(f"Min / Max SOC:               {SOC_hist.min():.2f} / {SOC_hist.max():.2f}")
print(f"Mean cycle efficiency:       {eta_cycle[50:].mean():.3f}")
print(f"Mean thermal power:          {P_th[50:].mean():.2f} kWt")
print(f"Mean electrical power:       {P_elec[50:].mean():.2f} kWe")
print(f"Mean vehicle demand:         {P_demand.mean():.2f} kWe")
soc_violation = np.any(SOC_hist < SOC_min) or np.any(SOC_hist > SOC_max)
print(f"SOC bounds respected:        {not soc_violation}")

# -----------------------------------------------------------------------
# 8. FIGURES
# -----------------------------------------------------------------------
th = t / 60.0  # minutes

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(th, P_demand, label="Vehicle demand $P_{demand}$", color="#c0392b", lw=1.4)
ax.plot(th, P_elec, label="Reactor electrical output $P_e$", color="#2471a3", lw=1.4)
ax.plot(th, P_th, label="Core thermal power $P_{th}$", color="#7d3c98", lw=1.2, ls="--")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Power (kW)")
ax.set_title("Fig. 1 — Reactor Power Coupled to X-01 Transition-Duty-Cycle Demand")
ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig1_power_coupling.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(th, T_core, label="Core temperature $T_{core}$", color="#c0392b")
ax.plot(th, T_out, label="Coolant outlet $T_{out}$", color="#1a5276")
ax.axhline(T_max_core, color="k", ls=":", label="Design limit $T_{max}$")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Temperature (K)")
ax.set_title("Fig. 2 — Core and Coolant Thermal Response")
ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig2_thermal_response.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(th, eta_cycle * 100, color="#117864")
ax.set_xlabel("Time (min)"); ax.set_ylabel("Brayton cycle efficiency (%)")
ax.set_title("Fig. 3 — Recuperated Closed-Brayton Conversion Efficiency")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig3_cycle_efficiency.png", dpi=160)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(th, SOC_hist * 100, color="#b9770e")
ax.axhline(SOC_min * 100, color="k", ls=":", lw=1)
ax.axhline(SOC_max * 100, color="k", ls=":", lw=1)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Buffer state of charge (%)")
ax.set_title("Fig. 4 — Energy-Buffer State of Charge Under Transition Loading")
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("fig4_soc.png", dpi=160)

# -----------------------------------------------------------------------
# 9. MASS-OPTIMIZATION STUDY: reactor rating vs. buffer capacity
# -----------------------------------------------------------------------
# Simple representative mass scaling (order-of-magnitude, open-literature
# style scaling laws for compact space/marine reactor power systems):
#   m_reactor  = m0_reactor + k_reactor * P_th_rating         (kg)
#   m_buffer   = k_buffer * E_buffer_capacity                  (kg)
# with a feasibility constraint: SOC must stay within [SOC_min, SOC_max]
# and T_core must stay below T_max_core across the mission profile.

m0_reactor = 220.0      # kg, fixed shielding/structure/PCU baseline mass
k_reactor = 9.5          # kg per kWt of rating
k_buffer = 7.0            # kg per kWh of buffer capacity

ratings = np.arange(20, 70, 2.5)
buffers = np.arange(2, 14, 1.0)
mass_grid = np.full((len(ratings), len(buffers)), np.nan)
feasible = np.zeros_like(mass_grid, dtype=bool)

def run_quick(P_rating, E_buf):
    """Fast re-run of the core/buffer loop for the optimization sweep
    (identical physics to the main loop, vectorization skipped for
    clarity at this project level)."""
    P_th_l = np.zeros(n); T_core_l = np.zeros(n); SOC_l = np.zeros(n)
    P_th_l[0] = 20.0; T_core_l[0] = 900.0; SOC_l[0] = SOC_setpoint
    for i in range(1, n):
        soc_err = SOC_setpoint - SOC_l[i - 1]
        Pc = np.clip(15.0 + Kp_soc * soc_err, 3.0, P_rating)
        P_th_l[i] = P_th_l[i - 1] + (Pc - P_th_l[i - 1]) / tau_core * dt
        Qr = m_dot * cp_coolant * (T_core_l[i - 1] - T_in) / 1000.0
        Ql = UA_loss * (T_core_l[i - 1] - T_ambient) / 1000.0
        T_core_l[i] = T_core_l[i - 1] + (P_th_l[i] - Qr - Ql) * 1000.0 / C_core * dt
        eta = brayton_efficiency(T_core_l[i])
        Pe = eta * P_th_l[i] * eta_generator
        net = (Pe - P_demand[i]) * dt / 3600.0
        net = net * eta_battery_rt if net > 0 else net
        SOC_l[i] = np.clip(SOC_l[i - 1] + net / E_buf, 0.0, 1.0)
    return T_core_l.max(), SOC_l.min(), SOC_l.max()

for a, Pr in enumerate(ratings):
    for b, Eb in enumerate(buffers):
        Tmax_i, SOCmin_i, SOCmax_i = run_quick(Pr, Eb)
        ok = (Tmax_i <= T_max_core) and (SOCmin_i >= SOC_min) and (SOCmax_i <= SOC_max)
        feasible[a, b] = ok
        mass = m0_reactor + k_reactor * Pr + k_buffer * Eb
        mass_grid[a, b] = mass if ok else np.nan

fig, ax = plt.subplots(figsize=(7.5, 5.5))
im = ax.pcolormesh(buffers, ratings, mass_grid, shading="auto", cmap="viridis")
cb = fig.colorbar(im, ax=ax); cb.set_label("System mass (kg)")
best_Pr = best_Eb = best_m = None
if np.any(feasible):
    idx = np.unravel_index(np.nanargmin(mass_grid), mass_grid.shape)
    best_Pr, best_Eb, best_m = ratings[idx[0]], buffers[idx[1]], mass_grid[idx]
    ax.scatter([best_Eb], [best_Pr], color="red", marker="*", s=180,
               label=f"Min-mass feasible point\n{best_Pr:.1f} kWt, {best_Eb:.1f} kWh, {best_m:.0f} kg")
    ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("Buffer capacity (kWh)")
ax.set_ylabel("Reactor thermal rating (kWt)")
ax.set_title("Fig. 5 — Feasible-Design Mass Map (Reactor Rating vs. Buffer Capacity)")
fig.tight_layout(); fig.savefig("fig5_mass_optimization.png", dpi=160)

if best_Pr is not None:
    print(f"\nOptimization result: minimum-mass FEASIBLE design = "
          f"{best_Pr:.1f} kWt reactor + {best_Eb:.1f} kWh buffer -> {best_m:.0f} kg total")
else:
    print("\nOptimization result: NO feasible design found in the swept grid "
          "(widen ratings/buffers ranges).")

np.savez("sim_results.npz", t=t, P_demand=P_demand, P_th=P_th, P_elec=P_elec,
         T_core=T_core, T_out=T_out, eta_cycle=eta_cycle, SOC=SOC_hist)
print("\nSaved: fig1..fig5 PNGs, sim_results.npz")
