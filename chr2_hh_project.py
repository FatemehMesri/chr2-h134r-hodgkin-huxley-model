import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

OUTPUT_DIR = Path(__file__).resolve().parent / "chr2_hh_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CHR2 = {
    "gmax": 0.4,
    "E": 0.0,
    "gamma": 0.1,
    "A": 10.6408,
    "B": -14.6408,
    "C": -42.7671,
    "wavelength_nm": 470.0,
    "hc": 1.986446e-25,
    "wloss": 1.3,
    "sigma_retinal": 12.0e-20,
    "tau_p": 1.3,
    "reference_temperature": 22.0,
    "Q10_Gd1": 1.97,
    "Q10_Gd2": 1.77,
    "Q10_Gr": 2.56,
    "Q10_e12dark": 1.10,
    "Q10_e21dark": 1.95,
    "Q10_epsilon1": 1.46,
    "Q10_epsilon2": 2.77,
}

HH = {
    "Cm": 1.0,
    "gNa": 120.0,
    "gK": 36.0,
    "gL": 0.3,
    "ENa": 50.0,
    "EK": -77.0,
    "EL": -54.387,
    "reference_temperature": 6.3,
    "Q10": 3.0,
}


def thermal_scale(value, q10, temperature, reference_temperature):
    return value * q10 ** ((temperature - reference_temperature) / 10.0)


def photon_flux_per_ms(irradiance):
    irradiance = max(float(irradiance), 0.0)
    if irradiance == 0.0:
        return 0.0
    photon_energy = 1.0e9 * CHR2["hc"] / CHR2["wavelength_nm"]
    photon_flux_per_second = 1000.0 * irradiance / photon_energy
    return photon_flux_per_second * CHR2["sigma_retinal"] / (CHR2["wloss"] * 1000.0)


def chr2_rates(voltage, irradiance, temperature=22.0):
    irradiance = max(float(irradiance), 0.0)
    temperature_delta = temperature - CHR2["reference_temperature"]
    gd1_22 = 0.075 + 0.043 * np.tanh(-(voltage + 20.0) / 20.0)
    gd2_22 = 0.05
    gr_22 = 0.0000434587 * np.exp(np.clip(-0.0211539274 * voltage, -100.0, 100.0))
    e12_dark_22 = 0.011
    e21_dark_22 = 0.008
    eps1_22 = 0.8535
    eps2_22 = 0.14
    gd1 = thermal_scale(gd1_22, CHR2["Q10_Gd1"], temperature, CHR2["reference_temperature"])
    gd2 = thermal_scale(gd2_22, CHR2["Q10_Gd2"], temperature, CHR2["reference_temperature"])
    gr = thermal_scale(gr_22, CHR2["Q10_Gr"], temperature, CHR2["reference_temperature"])
    e12_dark = thermal_scale(e12_dark_22, CHR2["Q10_e12dark"], temperature, CHR2["reference_temperature"])
    e21_dark = thermal_scale(e21_dark_22, CHR2["Q10_e21dark"], temperature, CHR2["reference_temperature"])
    eps1 = thermal_scale(eps1_22, CHR2["Q10_epsilon1"], temperature, CHR2["reference_temperature"])
    eps2 = thermal_scale(eps2_22, CHR2["Q10_epsilon2"], temperature, CHR2["reference_temperature"])
    log_light = np.log1p(irradiance / 0.024) if irradiance > 0.0 else 0.0
    e12 = e12_dark + 0.005 * log_light
    e21 = e21_dark + 0.004 * log_light
    s0 = 0.5 * (1.0 + np.tanh(120.0 * (100.0 * irradiance - 0.1)))
    photon_flux = photon_flux_per_ms(irradiance)
    return gd1, gd2, gr, e12, e21, eps1, eps2, photon_flux, s0


def chr2_state_derivatives(voltage, irradiance, states, temperature=22.0):
    o1, o2, c1, p = states
    c2 = 1.0 - c1 - o1 - o2
    gd1, gd2, gr, e12, e21, eps1, eps2, photon_flux, s0 = chr2_rates(voltage, irradiance, temperature)
    light_gate = photon_flux * p
    do1 = -(gd1 + e12) * o1 + e21 * o2 + eps1 * light_gate * c1
    do2 = -(gd2 + e21) * o2 + e12 * o1 + eps2 * light_gate * c2
    dc1 = -eps1 * light_gate * c1 + gd1 * o1 + gr * c2
    dp = (s0 - p) / CHR2["tau_p"]
    return np.array([do1, do2, dc1, dp])


def chr2_current_mA(voltage, states):
    o1, o2 = states[0], states[1]
    open_fraction = o1 + CHR2["gamma"] * o2
    exponent = np.exp(np.clip(voltage / CHR2["C"], -100.0, 100.0))
    numerator = CHR2["A"] + CHR2["B"] * exponent
    if abs(CHR2["E"]) < 1.0e-12:
        current_uA = CHR2["gmax"] * numerator * open_fraction
    else:
        if abs(voltage) < 1.0e-8:
            raise ValueError("The published empirical G(V) expression is singular at V=0 when EChR2 is nonzero.")
        rectification = numerator / voltage
        current_uA = CHR2["gmax"] * rectification * open_fraction * (voltage - CHR2["E"])
    return 0.001 * current_uA


def light_train(start, frequency, count, width, intensity):
    period = 1000.0 / frequency
    return [(start + index * period, width, intensity) for index in range(count)]


def solve_piecewise(rhs_factory, initial_state, duration, pulses, output_dt=0.05, max_step=0.05, rtol=1.0e-7, atol=1.0e-9):
    boundaries = {0.0, float(duration)}
    for start, width, intensity in pulses:
        if 0.0 < start < duration:
            boundaries.add(float(start))
        end = start + width
        if 0.0 < end < duration:
            boundaries.add(float(end))
    boundaries = sorted(boundaries)
    all_times = []
    all_states = []
    state = np.asarray(initial_state, dtype=float)
    for segment_index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        midpoint = 0.5 * (left + right)
        irradiance = sum(intensity for start, width, intensity in pulses if start <= midpoint < start + width)
        rhs = rhs_factory(irradiance)
        segment_times = np.arange(left, right, output_dt)
        if segment_times.size == 0 or segment_times[0] != left:
            segment_times = np.insert(segment_times, 0, left)
        segment_times = np.append(segment_times[segment_times < right], right)
        solution = solve_ivp(
            rhs,
            (left, right),
            state,
            method="LSODA",
            t_eval=segment_times,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        if segment_index == 0:
            all_times.extend(solution.t)
            all_states.extend(solution.y.T)
        else:
            all_times.extend(solution.t[1:])
            all_states.extend(solution.y.T[1:])
        state = solution.y[:, -1]
    return np.asarray(all_times), np.asarray(all_states)


def simulate_voltage_clamp(voltage=-70.0, intensity=5.0, width=100.0, delay=50.0, duration=250.0, temperature=22.0, output_dt=0.02):
    pulses = [(delay, width, intensity)]
    initial_state = np.array([0.0, 0.0, 1.0, 0.0])
    rhs_factory = lambda irradiance: lambda t, state: chr2_state_derivatives(voltage, irradiance, state, temperature)
    times, states = solve_piecewise(rhs_factory, initial_state, duration, pulses, output_dt=output_dt, max_step=0.02)
    current_mA = np.array([chr2_current_mA(voltage, state) for state in states])
    return times, states, current_mA, pulses


def vtrap(x, scale):
    z = x / scale
    if abs(z) < 1.0e-7:
        return scale * (1.0 + 0.5 * z)
    return x / (-np.expm1(-z))


def hh_gate_rates(voltage, temperature=22.0):
    alpha_m = 0.1 * vtrap(voltage + 40.0, 10.0)
    beta_m = 4.0 * np.exp(np.clip(-(voltage + 65.0) / 18.0, -100.0, 100.0))
    alpha_h = 0.07 * np.exp(np.clip(-(voltage + 65.0) / 20.0, -100.0, 100.0))
    beta_h = 1.0 / (1.0 + np.exp(np.clip(-(voltage + 35.0) / 10.0, -100.0, 100.0)))
    alpha_n = 0.01 * vtrap(voltage + 55.0, 10.0)
    beta_n = 0.125 * np.exp(np.clip(-(voltage + 65.0) / 80.0, -100.0, 100.0))
    factor = HH["Q10"] ** ((temperature - HH["reference_temperature"]) / 10.0)
    return tuple(factor * rate for rate in (alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n))


def hh_initial_state(voltage=-65.0):
    alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = hh_gate_rates(voltage, HH["reference_temperature"])
    m = alpha_m / (alpha_m + beta_m)
    h = alpha_h / (alpha_h + beta_h)
    n = alpha_n / (alpha_n + beta_n)
    return np.array([voltage, m, h, n, 0.0, 0.0, 1.0, 0.0])


def hh_currents(voltage, m, h, n, chr2_states):
    sodium = HH["gNa"] * m**3 * h * (voltage - HH["ENa"])
    potassium = HH["gK"] * n**4 * (voltage - HH["EK"])
    leak = HH["gL"] * (voltage - HH["EL"])
    chr2 = 1000.0 * chr2_current_mA(voltage, chr2_states)
    return sodium, potassium, leak, chr2


def simulate_hh(pulses, duration, temperature=22.0, applied_current=0.0, output_dt=0.05):
    initial_state = hh_initial_state()

    def rhs_factory(irradiance):
        def rhs(t, state):
            voltage, m, h, n = state[:4]
            chr2_states = state[4:8]
            alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n = hh_gate_rates(voltage, temperature)
            dm = alpha_m * (1.0 - m) - beta_m * m
            dh = alpha_h * (1.0 - h) - beta_h * h
            dn = alpha_n * (1.0 - n) - beta_n * n
            sodium, potassium, leak, chr2 = hh_currents(voltage, m, h, n, chr2_states)
            dv = (applied_current - sodium - potassium - leak - chr2) / HH["Cm"]
            d_chr2 = chr2_state_derivatives(voltage, irradiance, chr2_states, temperature)
            return np.array([dv, dm, dh, dn, *d_chr2])
        return rhs

    return solve_piecewise(rhs_factory, initial_state, duration, pulses, output_dt=output_dt, max_step=output_dt)


def detect_spikes(times, voltage, threshold=0.0):
    crossings = np.where((voltage[:-1] < threshold) & (voltage[1:] >= threshold))[0]
    spike_times = []
    for index in crossings:
        crossing_time = times[index] + (threshold - voltage[index]) * (times[index + 1] - times[index]) / (voltage[index + 1] - voltage[index])
        spike_times.append(crossing_time)
    return np.asarray(spike_times)


def response_to_single_pulse(intensity, width, temperature=22.0, delay=20.0, recovery=100.0, output_dt=0.1):
    duration = delay + width + recovery
    pulses = [(delay, width, intensity)]
    times, states = simulate_hh(pulses, duration, temperature=temperature, output_dt=output_dt)
    spike_times = detect_spikes(times, states[:, 0])
    after_onset = spike_times[spike_times >= delay]
    latency = float(after_onset[0] - delay) if after_onset.size else np.nan
    return times, states, spike_times, latency


def estimate_threshold(width, temperature=22.0, lower=0.0, upper=5.5, iterations=11):
    _, _, low_spikes, _ = response_to_single_pulse(lower, width, temperature)
    _, _, high_spikes, _ = response_to_single_pulse(upper, width, temperature)
    if low_spikes.size > 0:
        return lower
    if high_spikes.size == 0:
        return np.nan
    for _ in range(iterations):
        middle = 0.5 * (lower + upper)
        _, _, middle_spikes, _ = response_to_single_pulse(middle, width, temperature)
        if middle_spikes.size:
            upper = middle
        else:
            lower = middle
    return upper


def summarize_voltage_clamp(times, current_mA, pulse):
    start, width, _ = pulse
    stop = start + width
    magnitude = np.abs(current_mA)
    baseline = float(np.mean(magnitude[times < start])) if np.any(times < start) else 0.0
    during = np.where((times >= start) & (times <= stop))[0]
    after = np.where(times > stop)[0]
    peak_index = during[np.argmax(magnitude[during])]
    peak = float(magnitude[peak_index])
    steady_start = start + 0.8 * width
    steady_indices = np.where((times >= steady_start) & (times <= stop))[0]
    steady = float(np.mean(magnitude[steady_indices])) if steady_indices.size else np.nan
    rise_target = baseline + 0.6321205588 * (peak - baseline)
    rise_indices = np.where((times >= start) & (times <= times[peak_index]) & (magnitude >= rise_target))[0]
    tau_on = float(times[rise_indices[0]] - start) if rise_indices.size else np.nan
    inactivation_target = steady + (peak - steady) / np.e
    inactivation_indices = np.where((times > times[peak_index]) & (times <= stop) & (magnitude <= inactivation_target))[0]
    tau_inact = float(times[inactivation_indices[0]] - times[peak_index]) if inactivation_indices.size else np.nan
    tau_off = np.nan
    if after.size:
        off_initial = float(magnitude[np.argmin(np.abs(times - stop))])
        off_target = baseline + (off_initial - baseline) / np.e
        off_indices = after[magnitude[after] <= off_target]
        if off_indices.size:
            tau_off = float(times[off_indices[0]] - stop)
    return {
        "peak_mA_cm2": float(np.max(magnitude[during])),
        "steady_mA_cm2": steady,
        "tau_on_ms_approx": tau_on,
        "tau_inact_ms_approx": tau_inact,
        "tau_off_ms_approx": tau_off,
    }


def pulse_train_metrics(frequency, intensity, width, count=8, temperature=22.0):
    start = 20.0
    pulses = light_train(start, frequency, count, width, intensity)
    duration = pulses[-1][0] + 1000.0 / frequency
    times, states = simulate_hh(pulses, duration, temperature=temperature, output_dt=0.1)
    spike_times = detect_spikes(times, states[:, 0])
    latencies = []
    for index, (pulse_start, pulse_width, pulse_intensity) in enumerate(pulses):
        next_start = pulses[index + 1][0] if index + 1 < len(pulses) else duration
        candidates = spike_times[(spike_times >= pulse_start) & (spike_times < next_start)]
        latencies.append(float(candidates[0] - pulse_start) if candidates.size else np.nan)
    valid = np.asarray(latencies)[np.isfinite(latencies)]
    within_train_latency_sd = float(np.std(valid, ddof=1)) if valid.size > 1 else np.nan
    return {
        "frequency_Hz": frequency,
        "spike_count": int(spike_times.size),
        "pulse_count": count,
        "spikes_per_pulse": float(spike_times.size / count),
        "latency_sd_within_train_ms": within_train_latency_sd,
        "valid_pulse_latencies": int(valid.size),
        "times": times,
        "states": states,
        "spike_times": spike_times,
        "latencies": np.asarray(latencies),
    }


def save_csv(filename, rows):
    if not rows:
        return
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_project(temperature=22.0):
    clamp_times, clamp_states, clamp_current, clamp_pulses = simulate_voltage_clamp(
        voltage=-70.0, intensity=5.0, width=100.0, delay=50.0, duration=250.0, temperature=temperature
    )
    kinetics = summarize_voltage_clamp(clamp_times, clamp_current, clamp_pulses[0])

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(clamp_times, 1000.0 * clamp_current, linewidth=1.8)
    axis.axvspan(50.0, 150.0, alpha=0.15)
    axis.set(xlabel="Time (ms)", ylabel="I_ChR2 (µA/cm²)", title="ChR2-H134R voltage clamp: 5 mW/mm², 100 ms, V = −70 mV")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "01_voltage_clamp_trace.png", dpi=180)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    for intensity in (0.34, 1.0, 5.5):
        times, states, current, pulses = simulate_voltage_clamp(
            voltage=-70.0, intensity=intensity, width=100.0, delay=30.0, duration=180.0, temperature=temperature
        )
        axis.plot(times, 1000.0 * current, label=f"{intensity:g} mW/mm²")
    axis.axvspan(30.0, 130.0, alpha=0.12)
    axis.set(xlabel="Time (ms)", ylabel="I_ChR2 (µA/cm²)", title="Irradiance dependence under voltage clamp")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "02_irradiance_voltage_clamp.png", dpi=180)

    voltage_values = np.array([-80.0, -60.0, -40.0, -20.0, -10.0])
    iv_rows = []
    for voltage in voltage_values:
        times, states, current, pulses = simulate_voltage_clamp(
            voltage=voltage, intensity=5.5, width=100.0, delay=20.0, duration=150.0,
            temperature=temperature, output_dt=0.05
        )
        during = (times >= 20.0) & (times <= 120.0)
        peak_current = float(np.min(current[during]))
        steady_current = float(np.mean(current[(times >= 100.0) & (times <= 120.0)]))
        iv_rows.append({"voltage_mV": float(voltage), "peak_current_mA_cm2": peak_current,
                        "steady_current_mA_cm2": steady_current})
    save_csv("voltage_current_relation.csv", iv_rows)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["voltage_mV"] for row in iv_rows], [1000.0 * row["peak_current_mA_cm2"] for row in iv_rows], marker="o", label="Peak")
    axis.plot([row["voltage_mV"] for row in iv_rows], [1000.0 * row["steady_current_mA_cm2"] for row in iv_rows], marker="s", label="Steady state")
    axis.set(xlabel="Holding voltage (mV)", ylabel="I_ChR2 (µA/cm²)", title="ChR2 current–voltage relation at 5.5 mW/mm²")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "08_current_voltage_relation.png", dpi=180)

    kinetic_rows = []
    for intensity in (0.34, 1.0, 2.0, 5.5):
        times, states, current, pulses = simulate_voltage_clamp(
            voltage=-70.0, intensity=intensity, width=100.0, delay=30.0,
            duration=220.0, temperature=temperature, output_dt=0.02
        )
        metrics = summarize_voltage_clamp(times, current, pulses[0])
        kinetic_rows.append({"irradiance_mW_mm2": intensity, **metrics})
    save_csv("kinetic_metrics.csv", kinetic_rows)

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, key, ylabel in zip(
        axes,
        ("tau_on_ms_approx", "tau_inact_ms_approx", "tau_off_ms_approx"),
        ("τON (ms)", "τINACT (ms)", "τOFF (ms)"),
    ):
        axis.plot([row["irradiance_mW_mm2"] for row in kinetic_rows], [row[key] for row in kinetic_rows], marker="o")
        axis.set(xlabel="Irradiance (mW/mm²)", ylabel=ylabel)
        axis.grid(alpha=0.25)
    figure.suptitle("Approximate ChR2 kinetic time scales")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "09_kinetic_time_scales.png", dpi=180)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(clamp_times, clamp_states[:, 0], label="O1")
    axis.plot(clamp_times, clamp_states[:, 1], label="O2")
    axis.plot(clamp_times, clamp_states[:, 2], label="C1")
    axis.plot(clamp_times, 1.0 - np.sum(clamp_states[:, :3], axis=1), label="C2")
    axis.plot(clamp_times, clamp_states[:, 3], label="p", linestyle="--")
    axis.set(xlabel="Time (ms)", ylabel="State fraction / activation variable", title="ChR2 photocycle state occupancy")
    axis.legend(ncol=3, frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "03_state_occupancy.png", dpi=180)

    example_pulses = [(20.0, 10.0, 5.5)]
    hh_times, hh_states = simulate_hh(example_pulses, 180.0, temperature=temperature, output_dt=0.025)
    hh_spikes = detect_spikes(hh_times, hh_states[:, 0])
    hh_ch_current = np.array([1000.0 * chr2_current_mA(row[0], row[4:8]) for row in hh_states])
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(hh_times, hh_states[:, 0], linewidth=1.5)
    axes[0].axvspan(20.0, 30.0, alpha=0.15)
    axes[0].set(ylabel="V (mV)", title="Hodgkin–Huxley membrane driven by ChR2-H134R")
    axes[1].plot(hh_times, hh_ch_current, linewidth=1.3)
    axes[1].axvspan(20.0, 30.0, alpha=0.15)
    axes[1].set(xlabel="Time (ms)", ylabel="I_ChR2 (µA/cm²)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "04_hh_optical_spike.png", dpi=180)

    intensities = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.5])
    intensity_rows = []
    for intensity in intensities:
        times, states, spikes, latency = response_to_single_pulse(intensity, 10.0, temperature)
        intensity_rows.append({
            "irradiance_mW_mm2": float(intensity),
            "spike_count": int(spikes.size),
            "first_spike_latency_ms": latency,
            "peak_voltage_mV": float(np.max(states[:, 0])),
        })
    save_csv("intensity_sweep.csv", intensity_rows)

    threshold_10ms = estimate_threshold(10.0, temperature)
    widths = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    strength_duration_rows = []
    for width in widths:
        threshold = estimate_threshold(float(width), temperature, iterations=8)
        strength_duration_rows.append({"pulse_width_ms": float(width), "threshold_mW_mm2": float(threshold)})
    save_csv("strength_duration.csv", strength_duration_rows)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot([row["pulse_width_ms"] for row in strength_duration_rows],
              [row["threshold_mW_mm2"] for row in strength_duration_rows], marker="o")
    axis.set(xlabel="Pulse width (ms)", ylabel="Threshold irradiance (mW/mm²)", title="Optical strength–duration relation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "05_strength_duration.png", dpi=180)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot([row["irradiance_mW_mm2"] for row in intensity_rows], [row["spike_count"] for row in intensity_rows], marker="o")
    axes[0].set(ylabel="Spike count", title="Intensity-dependent excitability")
    axes[1].plot([row["irradiance_mW_mm2"] for row in intensity_rows], [row["first_spike_latency_ms"] for row in intensity_rows], marker="o")
    axes[1].set(xlabel="Irradiance (mW/mm²)", ylabel="First-spike latency (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "06_single_pulse_intensity_response.png", dpi=180)

    constant_intensity_rows = []
    for intensity in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.5):
        continuous_pulses = [(0.0, 300.0, intensity)] if intensity > 0.0 else []
        constant_times, constant_states = simulate_hh(
            continuous_pulses, 300.0, temperature=temperature, output_dt=0.1
        )
        constant_spikes = detect_spikes(constant_times, constant_states[:, 0])
        constant_intensity_rows.append({
            "constant_irradiance_mW_mm2": float(intensity),
            "spike_count_300ms": int(constant_spikes.size),
            "mean_rate_Hz": float(1000.0 * constant_spikes.size / 300.0),
            "maximum_voltage_mV": float(np.max(constant_states[:, 0])),
        })
    save_csv("constant_illumination_bifurcation.csv", constant_intensity_rows)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["constant_irradiance_mW_mm2"] for row in constant_intensity_rows],
        [row["mean_rate_Hz"] for row in constant_intensity_rows], marker="o"
    )
    axis.set(xlabel="Constant irradiance (mW/mm²)", ylabel="Mean spike rate over 300 ms (Hz)",
             title="Onset of optical excitability under sustained illumination")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "10_constant_illumination_response.png", dpi=180)

    suprathreshold_intensity = max(1.0, 1.5 * threshold_10ms) if np.isfinite(threshold_10ms) else 2.0
    frequencies = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    frequency_rows = []
    frequency_details = []
    for frequency in frequencies:
        result = pulse_train_metrics(float(frequency), suprathreshold_intensity, 5.0, count=6, temperature=temperature)
        frequency_rows.append({key: result[key] for key in (
            "frequency_Hz", "spike_count", "pulse_count", "spikes_per_pulse",
            "latency_sd_within_train_ms", "valid_pulse_latencies"
        )})
        frequency_details.append(result)
    save_csv("frequency_sweep.csv", frequency_rows)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot([row["frequency_Hz"] for row in frequency_rows], [row["spikes_per_pulse"] for row in frequency_rows], marker="o")
    axes[0].set(ylabel="Spikes per light pulse", title=f"Pulse-train following ({suprathreshold_intensity:.3g} mW/mm², 5 ms pulses)")
    axes[1].plot([row["frequency_Hz"] for row in frequency_rows], [row["latency_sd_within_train_ms"] for row in frequency_rows], marker="o")
    axes[1].set(xlabel="Pulse frequency (Hz)", ylabel="Within-train latency SD (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "07_frequency_and_timing.png", dpi=180)

    print(f"Temperature: {temperature:.1f} °C")
    print("Voltage-clamp metrics:")
    for name, value in kinetics.items():
        print(f"  {name}: {value:.6g}")
    print(f"HH threshold for a 10 ms pulse: {threshold_10ms:.6g} mW/mm²")
    print(f"HH example spikes: {hh_spikes.size}")
    print("Strength-duration results:")
    for row in strength_duration_rows:
        print(f"  {row['pulse_width_ms']:>5.1f} ms : {row['threshold_mW_mm2']:.6g} mW/mm²")
    print("Frequency results:")
    for row in frequency_rows:
        print(f"  {row['frequency_Hz']:>5.1f} Hz : {row['spike_count']} spikes, {row['spikes_per_pulse']:.3g} spikes/pulse, latency SD={row['latency_sd_within_train_ms']}")
    print(f"Figures and CSV files were saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_project(temperature=22.0)
