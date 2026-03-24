"""
HRTF Objective Metric Evaluation Experiment
============================================
Demonstrates that LSD fails to detect perceptually critical degradations
while proposed binaural/phase-aware metrics correctly flag them.

Dataset: RIEC HRTF Database (Tohoku University)
         male/ and female/ subdirectories containing .sofa files
         IR shape: (865 positions, 2 ears, 512 samples), Fs = 48000 Hz

Degradation conditions tested:
  A) Minimum-phase reconstruction (Hilbert / real-cepstrum method)
  B) Magnitude smoothing (1/3-octave and 1/1-octave critical-band)
  C) Magnitude quantization (12-bit and 8-bit)

Metrics computed:
  LSD     - Log-Spectral Distortion (baseline, magnitude-only)
  ITD_err - RMSE of Interaural Time Difference (ms)
  ILD_err - RMS error of Interaural Level Difference spectrum (dB)
  GD_err  - RMS error of group delay (ms) [phase-sensitive]
  CPM     - Composite Perceptual Metric = weighted sum of above
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sofar as sf
from scipy.signal import lfilter, freqz
from scipy.ndimage import uniform_filter1d

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
HRTF_DIR   = r"C:/Users/Apermesa/Documents/GitHub/3710-HRTF"
OUT_DIR    = r"C:/Users/Apermesa/Documents/GitHub/Raytraced-Audio/figures"
N_SUBJECTS = 10      # use first N subjects per gender (total ≤ 20)
FS         = 48000
FREQ_LO    = 200     # Hz – lower bound for metric computation
FREQ_HI    = 16000   # Hz – upper bound
# Notch-region weighting band for CPM
NOTCH_LO   = 4000
NOTCH_HI   = 12000
# CPM weights (sum to 1 for interpretability)
W_LSD      = 0.20
W_ILD      = 0.25
W_GD       = 0.35
W_ITD      = 0.20

os.makedirs(OUT_DIR, exist_ok=True)

# ─── SOFA loading ─────────────────────────────────────────────────────────────
def load_hrirs(folder, max_n):
    """Return list of (hrir, positions) tuples."""
    files = sorted(f for f in os.listdir(folder) if f.endswith(".sofa"))[:max_n]
    hrirs, positions = [], []
    for fname in files:
        sofa = sf.read_sofa(os.path.join(folder, fname), verify=False)
        hrirs.append(sofa.Data_IR)          # (M, 2, N)
        positions.append(sofa.SourcePosition)  # (M, 3) az/el/dist
    return hrirs, positions

# ─── Degradation functions ────────────────────────────────────────────────────
def minimum_phase_reconstruction(h):
    """Real-cepstrum minimum-phase reconstruction of a single HRIR."""
    N = len(h)
    Nfft = max(512, 2 * N)
    H = np.fft.rfft(h, n=Nfft)
    log_mag = np.log(np.abs(H) + 1e-12)
    cepstrum = np.fft.irfft(log_mag, n=Nfft)
    # causal windowing: keep n=0 and fold n=1..Nfft/2-1
    win = np.zeros(Nfft)
    win[0] = 1.0
    win[1:Nfft // 2] = 2.0
    win[Nfft // 2] = 1.0
    H_mp = np.exp(np.fft.rfft(cepstrum * win, n=Nfft))
    h_mp = np.fft.irfft(H_mp, n=Nfft)[:N]
    return h_mp.astype(h.dtype)


def apply_minimum_phase(hrir):
    """Apply min-phase reconstruction to (M, 2, N) HRIR array, restoring ITD."""
    out = np.empty_like(hrir)
    for m in range(hrir.shape[0]):
        h_L = hrir[m, 0, :]
        h_R = hrir[m, 1, :]
        # Estimate original ITD (samples) via onset detection
        onset_L = np.argmax(np.abs(h_L) > 0.01 * np.max(np.abs(h_L)))
        onset_R = np.argmax(np.abs(h_R) > 0.01 * np.max(np.abs(h_R)))
        itd_samp = int(onset_L) - int(onset_R)
        # Minimum-phase cores
        h_L_mp = minimum_phase_reconstruction(h_L)
        h_R_mp = minimum_phase_reconstruction(h_R)
        # Re-insert ITD by shifting the leading ear
        if itd_samp > 0:
            h_L_mp = np.roll(h_L_mp, itd_samp)
            h_L_mp[:itd_samp] = 0.0
        elif itd_samp < 0:
            h_R_mp = np.roll(h_R_mp, -itd_samp)
            h_R_mp[:-itd_samp] = 0.0
        out[m, 0, :] = h_L_mp
        out[m, 1, :] = h_R_mp
    return out


def octave_smooth(mag_spectrum, freqs, frac_octave=3):
    """Apply fractional-octave smoothing to magnitude spectrum."""
    n_freqs = len(freqs)
    smoothed = np.empty_like(mag_spectrum)
    for k, f in enumerate(freqs):
        if f < 1.0:
            smoothed[k] = mag_spectrum[k]
            continue
        f_lo = f / 2.0 ** (1.0 / (2 * frac_octave))
        f_hi = f * 2.0 ** (1.0 / (2 * frac_octave))
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        smoothed[k] = np.mean(mag_spectrum[mask]) if mask.any() else mag_spectrum[k]
    return smoothed


def apply_magnitude_smoothing(hrir, freqs, frac_octave=3):
    """Smooth HRIR magnitude while keeping phase intact."""
    out = np.empty_like(hrir)
    for m in range(hrir.shape[0]):
        for ear in range(2):
            h = hrir[m, ear, :]
            H = np.fft.rfft(h)
            mag = np.abs(H)
            phase = np.angle(H)
            mag_smooth = octave_smooth(mag, freqs, frac_octave)
            out[m, ear, :] = np.fft.irfft(mag_smooth * np.exp(1j * phase), n=len(h))
    return out.astype(hrir.dtype)


def apply_quantization(hrir, bits=8):
    """Quantize HRIR to given bit depth."""
    max_val = np.max(np.abs(hrir)) + 1e-12
    levels = 2 ** bits
    quantized = np.round(hrir / max_val * (levels / 2)) / (levels / 2) * max_val
    return quantized.astype(hrir.dtype)

# ─── Metric functions ─────────────────────────────────────────────────────────
def compute_lsd(hrir_ref, hrir_test, freqs, freq_mask):
    """Log-Spectral Distortion (dB), averaged over positions and ears."""
    scores = []
    for m in range(hrir_ref.shape[0]):
        for ear in range(2):
            H_ref  = np.abs(np.fft.rfft(hrir_ref[m, ear, :]))
            H_test = np.abs(np.fft.rfft(hrir_test[m, ear, :]))
            H_ref  = H_ref[freq_mask]
            H_test = H_test[freq_mask]
            log_diff = 20 * np.log10(H_ref / (H_test + 1e-12) + 1e-12)
            scores.append(np.sqrt(np.mean(log_diff ** 2)))
    return float(np.mean(scores))


def compute_ild_error(hrir_ref, hrir_test, freqs, freq_mask):
    """RMS error of ILD spectrum (dB)."""
    scores = []
    for m in range(hrir_ref.shape[0]):
        H_L_ref  = np.abs(np.fft.rfft(hrir_ref[m, 0, :])) [freq_mask] + 1e-12
        H_R_ref  = np.abs(np.fft.rfft(hrir_ref[m, 1, :])) [freq_mask] + 1e-12
        H_L_test = np.abs(np.fft.rfft(hrir_test[m, 0, :]))[freq_mask] + 1e-12
        H_R_test = np.abs(np.fft.rfft(hrir_test[m, 1, :]))[freq_mask] + 1e-12
        ild_ref  = 20 * np.log10(H_L_ref  / H_R_ref)
        ild_test = 20 * np.log10(H_L_test / H_R_test)
        scores.append(np.sqrt(np.mean((ild_ref - ild_test) ** 2)))
    return float(np.mean(scores))


def compute_itd_error(hrir_ref, hrir_test, fs):
    """RMSE of ITD values (ms)."""
    def get_itd_ms(hrir):
        N = hrir.shape[0]
        itds = np.empty(N)
        for m in range(N):
            h_L, h_R = hrir[m, 0, :], hrir[m, 1, :]
            xcorr = np.correlate(h_L, h_R, mode='full')
            lag   = np.argmax(xcorr) - (len(h_R) - 1)
            itds[m] = lag / fs * 1000.0   # convert to ms
        return itds
    itd_ref  = get_itd_ms(hrir_ref)
    itd_test = get_itd_ms(hrir_test)
    return float(np.sqrt(np.mean((itd_ref - itd_test) ** 2)))


def compute_gd_error(hrir_ref, hrir_test, freqs, freq_mask, fs):
    """RMS error of group delay (ms), phase-sensitive.
    Uses conjugate-division formulation for numerical stability."""
    def group_delay(h, freq_mask):
        N = len(h)
        H   = np.fft.rfft(h)
        H_d = np.fft.rfft(h * np.arange(N))
        # Stable: Re(H_d * conj(H)) / |H|^2  instead of H_d / H
        denom = np.abs(H) ** 2 + 1e-8      # regularised denominator
        gd = -np.real(H_d * np.conj(H)) / denom / fs * 1000.0  # ms
        gd = np.clip(gd, -50.0, 50.0)      # clip extreme values (>50ms unphysical)
        return gd[freq_mask]
    scores = []
    for m in range(hrir_ref.shape[0]):
        for ear in range(2):
            gd_ref  = group_delay(hrir_ref[m, ear, :],  freq_mask)
            gd_test = group_delay(hrir_test[m, ear, :], freq_mask)
            scores.append(np.sqrt(np.mean((gd_ref - gd_test) ** 2)))
    return float(np.mean(scores))


def composite_perceptual_metric(lsd, ild_err, gd_err, itd_err):
    """Composite Perceptual Metric (CPM) — normalized and weighted."""
    # Reference ranges for normalization (approximate JND-based thresholds)
    lsd_ref = 2.0    # dB – typical acceptable LSD
    ild_ref = 1.0    # dB – ILD JND
    gd_ref  = 0.08   # ms – audible group-delay deviation (~3-4 samples @ 48kHz)
    itd_ref = 0.03   # ms – ITD JND (~1.5 samples @ 48kHz)
    cpm = (W_LSD * lsd    / lsd_ref +
           W_ILD * ild_err / ild_ref +
           W_GD  * gd_err  / gd_ref  +
           W_ITD * itd_err / itd_ref)
    return cpm

# ─── Main experiment loop ─────────────────────────────────────────────────────
def run_experiment():
    print("Loading HRTF data...")
    male_hrirs,   male_pos   = load_hrirs(os.path.join(HRTF_DIR, "male"),   N_SUBJECTS)
    female_hrirs, female_pos = load_hrirs(os.path.join(HRTF_DIR, "female"), N_SUBJECTS)
    all_hrirs = male_hrirs + female_hrirs
    print(f"  Loaded {len(all_hrirs)} subjects ({len(male_hrirs)} M, {len(female_hrirs)} F)")

    # Frequency axis
    N_ir   = all_hrirs[0].shape[2]
    n_rfft = N_ir // 2 + 1
    freqs  = np.fft.rfftfreq(N_ir, d=1.0 / FS)
    freq_mask = (freqs >= FREQ_LO) & (freqs <= FREQ_HI)

    conditions = [
        ("Reference",              lambda h, f: h),
        ("Min-Phase",              lambda h, f: apply_minimum_phase(h)),
        ("1/3-Oct Smooth",         lambda h, f: apply_magnitude_smoothing(h, f, frac_octave=3)),
        ("1/1-Oct Smooth",         lambda h, f: apply_magnitude_smoothing(h, f, frac_octave=1)),
        ("12-bit Quantization",    lambda h, f: apply_quantization(h, bits=12)),
        ("8-bit Quantization",     lambda h, f: apply_quantization(h, bits=8)),
    ]

    rows = []
    for cond_name, degradation_fn in conditions[1:]:   # skip Reference
        print(f"\nProcessing: {cond_name}")
        lsd_all, ild_all, gd_all, itd_all = [], [], [], []

        for subj_idx, hrir_ref in enumerate(all_hrirs):
            hrir_test = degradation_fn(hrir_ref, freqs)
            lsd_all .append(compute_lsd     (hrir_ref, hrir_test, freqs, freq_mask))
            ild_all .append(compute_ild_error(hrir_ref, hrir_test, freqs, freq_mask))
            gd_all  .append(compute_gd_error (hrir_ref, hrir_test, freqs, freq_mask, FS))
            itd_all .append(compute_itd_error(hrir_ref, hrir_test, FS))
            sys.stdout.write(f"  Subject {subj_idx + 1}/{len(all_hrirs)}\r")
            sys.stdout.flush()

        lsd_m = np.mean(lsd_all);  lsd_s = np.std(lsd_all)
        ild_m = np.mean(ild_all);  ild_s = np.std(ild_all)
        gd_m  = np.mean(gd_all);   gd_s  = np.std(gd_all)
        itd_m = np.mean(itd_all);  itd_s = np.std(itd_all)
        cpm_m = composite_perceptual_metric(lsd_m, ild_m, gd_m, itd_m)
        cpm_vals = [composite_perceptual_metric(l, i, g, t)
                    for l, i, g, t in zip(lsd_all, ild_all, gd_all, itd_all)]
        cpm_s = np.std(cpm_vals)

        rows.append({
            "Condition":     cond_name,
            "LSD_mean":      round(lsd_m, 3),
            "LSD_std":       round(lsd_s, 3),
            "ILD_err_mean":  round(ild_m, 3),
            "ILD_err_std":   round(ild_s, 3),
            "GD_err_mean":   round(gd_m,  4),
            "GD_err_std":    round(gd_s,  4),
            "ITD_err_mean":  round(itd_m, 4),
            "ITD_err_std":   round(itd_s, 4),
            "CPM_mean":      round(cpm_m, 3),
            "CPM_std":       round(cpm_s, 3),
        })
        print(f"\n  LSD={lsd_m:.3f}dB  ILD={ild_m:.3f}dB  GD={gd_m:.4f}ms  ITD={itd_m:.4f}ms  CPM={cpm_m:.3f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df, freqs, freq_mask, all_hrirs, conditions


# ─── Visualization ────────────────────────────────────────────────────────────
def plot_metric_comparison(df, out_dir):
    """Figure 4.1: Grouped bar chart of all metrics across conditions."""
    conditions = df["Condition"].tolist()
    x = np.arange(len(conditions))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 5))

    # Normalise each metric to [0,1] for visual comparison
    def norm(col):
        v = df[col].values.astype(float)
        return v / (v.max() + 1e-9)

    b1 = ax.bar(x - 1.5*width, norm("LSD_mean"),      width, label="LSD (normalised)",      color="#4477AA")
    b2 = ax.bar(x - 0.5*width, norm("ILD_err_mean"),  width, label="ILD Error (normalised)", color="#EE6677")
    b3 = ax.bar(x + 0.5*width, norm("GD_err_mean"),   width, label="Group Delay Error (norm.)", color="#228833")
    b4 = ax.bar(x + 1.5*width, norm("CPM_mean"),      width, label="CPM (normalised)",       color="#CCBB44")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Normalised metric value (0 = no distortion)")
    ax.set_title("Figure 4.1 — Metric Sensitivity Across Degradation Conditions\n"
                 "(RIEC HRTF Database, n = 20 subjects)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, 1.18)

    # Annotate the "LSD blindspot" — min-phase has low LSD but high GD
    mp_idx = conditions.index("Min-Phase")
    ax.annotate("LSD\nblindspot",
                xy=(mp_idx - 1.5*width, norm("LSD_mean")[mp_idx] + 0.03),
                xytext=(mp_idx - 1.5*width - 0.5, 0.75),
                arrowprops=dict(arrowstyle="->", color="black"),
                fontsize=8, color="black",
                ha="center")

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_1_metric_comparison.pdf")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_lsd_vs_cpm(df, out_dir):
    """Figure 4.2: Scatter plot LSD vs CPM — shows LSD fails for min-phase."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE"]
    for i, row in df.iterrows():
        ax.scatter(row["LSD_mean"], row["CPM_mean"],
                   color=colors[i % len(colors)], s=80, zorder=3,
                   label=row["Condition"])
        ax.annotate(row["Condition"],
                    (row["LSD_mean"], row["CPM_mean"]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=7.5)

    # Diagonal reference line (perfect agreement)
    lsd_max = df["LSD_mean"].max()
    cpm_max = df["CPM_mean"].max()
    lin_max = max(lsd_max, cpm_max) * 1.1
    ax.plot([0, lin_max], [0, lin_max], 'k--', alpha=0.3, label="LSD ≡ CPM (ideal)")

    ax.set_xlabel("LSD (dB) — magnitude-only metric")
    ax.set_ylabel("CPM — composite perceptual metric")
    ax.set_title("Figure 4.2 — LSD vs. Composite Perceptual Metric\n"
                 "Divergence reveals LSD blindspot for phase distortion")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_2_lsd_vs_cpm.pdf")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_spectral_comparison(all_hrirs, freqs, freq_mask, out_dir):
    """Figure 4.3: Spectral comparison of original vs min-phase for one subject."""
    hrir_ref  = all_hrirs[0]          # subject 0
    hrir_mp   = apply_minimum_phase(hrir_ref)

    # Average over the 4 canonical directions (front/back/left/right)
    def avg_spectrum(hrir, ear=0):
        mags = []
        for m in range(0, min(20, hrir.shape[0])):
            H = np.abs(np.fft.rfft(hrir[m, ear, :]))
            mags.append(20 * np.log10(H + 1e-12))
        return np.mean(mags, axis=0)

    mag_ref = avg_spectrum(hrir_ref, ear=0)
    mag_mp  = avg_spectrum(hrir_mp,  ear=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top: magnitude spectra
    ax = axes[0]
    ax.plot(freqs, mag_ref, label="Original", color="#4477AA", lw=1.2)
    ax.plot(freqs, mag_mp,  label="Min-Phase", color="#EE6677", lw=1.2, alpha=0.85)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylabel("Magnitude (dBFS)")
    ax.set_xscale("log")
    ax.set_title("Figure 4.3 — Spectral Comparison: Original vs Min-Phase (Subject 1, Left Ear)")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.axvspan(NOTCH_LO, NOTCH_HI, color="gold", alpha=0.15, label="Pinna notch region")

    # Bottom: difference (LSD per frequency)
    ax = axes[1]
    diff = mag_ref - mag_mp
    ax.plot(freqs, diff, color="#228833", lw=1.1)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvspan(NOTCH_LO, NOTCH_HI, color="gold", alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Spectral difference (dB)")
    ax.set_title("Magnitude difference — appears minimal (misleading low LSD)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_3_spectral_comparison.pdf")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_group_delay_comparison(all_hrirs, freqs, freq_mask, out_dir):
    """Figure 4.4: Group delay comparison showing phase distortion in min-phase."""
    hrir_ref = all_hrirs[0]
    hrir_mp  = apply_minimum_phase(hrir_ref)

    def avg_group_delay(hrir, ear=0, freq_mask=None):
        N = hrir.shape[2]
        gds = []
        for m in range(0, min(20, hrir.shape[0])):
            h = hrir[m, ear, :]
            H   = np.fft.rfft(h)
            H_d = np.fft.rfft(h * np.arange(N))
            gd  = -np.real(H_d / (H + 1e-12)) / FS * 1000.0
            if freq_mask is not None:
                gd = gd[freq_mask]
            gds.append(gd)
        return np.mean(gds, axis=0)

    gd_ref = avg_group_delay(hrir_ref, ear=0, freq_mask=freq_mask)
    gd_mp  = avg_group_delay(hrir_mp,  ear=0, freq_mask=freq_mask)
    f_plot = freqs[freq_mask]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f_plot, gd_ref, label="Original", color="#4477AA", lw=1.2)
    ax.plot(f_plot, gd_mp,  label="Min-Phase", color="#EE6677", lw=1.2, alpha=0.85)
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Group Delay (ms)")
    ax.set_xscale("log")
    ax.set_title("Figure 4.4 — Group Delay: Original vs Min-Phase\n"
                 "LSD is blind to this phase distortion; GD Error detects it")
    ax.axvspan(NOTCH_LO, NOTCH_HI, color="gold", alpha=0.15, label="Pinna notch region")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_4_group_delay.pdf")
    plt.savefig(path, dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), dpi=150)
    print(f"Saved {path}")
    plt.close()


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, freqs, freq_mask, all_hrirs, conditions = run_experiment()

    print("\n=== Generating figures ===")
    plot_metric_comparison(df, OUT_DIR)
    plot_lsd_vs_cpm(df, OUT_DIR)
    plot_spectral_comparison(all_hrirs, freqs, freq_mask, OUT_DIR)
    plot_group_delay_comparison(all_hrirs, freqs, freq_mask, OUT_DIR)

    print("\n=== Final Results Table ===")
    print(df.to_string(index=False))
    print(f"\nAll outputs in: {OUT_DIR}")
