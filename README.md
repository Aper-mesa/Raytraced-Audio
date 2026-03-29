# Objective HRTF Evaluation Metrics: Composite Perceptual Metric (CPM) Framework

**Author:** Shanghao Zou
**Supervisor:** Dr. Baha Ihnaini
**Institution:** Department of Computer Science and Technology, Wenzhou-Kean University
**Thesis Title:** Objective Head-Related Transfer Function Evaluation Metrics for More Consistent Perceptual Assessment

---

## Project Overview

This project investigates the failure modes of the industry-standard Log-Spectral Distortion (LSD) metric for Head-Related Transfer Function (HRTF) evaluation and proposes a **Composite Perceptual Metric (CPM)** that integrates four complementary objective measures:

1. **LSD** (Log-Spectral Distortion) -- magnitude-only baseline
2. **ILD Error** (Interaural Level Difference) -- binaural cue
3. **Group Delay Error** -- phase-sensitive metric
4. **ITD Error** (Interaural Time Difference) -- binaural timing cue

The central finding confirms the "LSD blindspot": minimum-phase reconstruction achieves the **lowest LSD** (0.030 dB) of all conditions yet produces the **highest CPM** (14.094) due to massive group delay error (2.335 ms), providing direct empirical support for the Andreopoulou & Katz (2022) hypothesis that magnitude-only metrics are fundamentally inadequate for HRTF quality prediction.

---

## Project Structure

```
HRTF-Evaluation/
|-- experiment.py              # Main experiment script (all metrics + plots)
|-- requirements.txt           # Python dependencies
|-- README.md                  # This file
|-- github_repository_link.txt # Link to public GitHub repository
|
|-- HRTF Dataset/              # RIEC HRTF Database (SOFA files)
|   |-- male/                  # 20 male subjects (.sofa)
|   |-- female/                # 11 female subjects (.sofa)
|
|-- figures/                   # Generated figures and results
|   |-- fig4_1_metric_comparison.png   # Grouped bar chart of all metrics
|   |-- fig4_2_lsd_vs_cpm.png         # Scatter plot: LSD vs CPM
|   |-- fig4_3_spectral_comparison.png # Spectral comparison (orig vs min-phase)
|   |-- fig4_4_group_delay.png         # Group delay comparison
|   |-- results.csv                    # Numerical results table
|
|-- screenshots/               # Screenshots of key experimental results
|   |-- (same figures as above, for documentation purposes)
|
|-- 1235425-W01-01-Project-proposal.tex  # Full thesis (LaTeX source)
|-- 1235425-W01-01-Project-proposal.pdf  # Compiled thesis PDF
|-- references.bib                        # Bibliography
```

---

## Environment Setup

### Prerequisites

- **Python 3.9 or later** (tested on Python 3.11)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Aper-mesa/Raytraced-Audio.git
cd Raytraced-Audio
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
| Package      | Version   | Purpose                                   |
|-------------|-----------|-------------------------------------------|
| numpy       | >= 1.24.0 | Numerical computation, FFT, array ops     |
| scipy       | >= 1.10.0 | Signal processing, cross-correlation      |
| pandas      | >= 2.0.0  | Tabular data management, CSV export       |
| matplotlib  | >= 3.7.0  | Visualization, figure generation          |
| sofar       | >= 1.1.0  | Reading SOFA-format HRTF files            |

---

## Dataset Setup

### RIEC HRTF Database

The experiment uses the **RIEC HRTF Database** from the Research Institute of Electrical Communication (RIEC) at Tohoku University.

- **Source:** https://www.ais.riec.tohoku.ac.jp/lab/db-hrtf/
- **Format:** SOFA (Spatially Oriented Format for Acoustics)
- **Specification:** 865 spatial positions, 2 ears, 512-sample impulse responses at 48 kHz

#### Download Instructions

1. Visit the RIEC HRTF Database website: https://www.ais.riec.tohoku.ac.jp/lab/db-hrtf/
2. Download the SOFA-format HRTF files for the desired subjects
3. Organize the files into the following directory structure:

```
HRTF Dataset/
|-- male/
|   |-- RIEC_hrir_subject_001.sofa
|   |-- RIEC_hrir_subject_002.sofa
|   |-- RIEC_hrir_subject_003.sofa
|   |-- ... (20 male subjects)
|
|-- female/
|   |-- RIEC_hrir_subject_004.sofa
|   |-- RIEC_hrir_subject_012.sofa
|   |-- RIEC_hrir_subject_018.sofa
|   |-- ... (11 female subjects)
```

> **Note:** The dataset is approximately 243 MB in total. The SOFA files are not included in the zip archive due to size constraints. Please download them directly from the RIEC website. The experiment uses the first 10 subjects from each gender directory (20 subjects total).

---

## Running the Experiment

### Step 1: Verify Dataset

Ensure the `HRTF Dataset/` directory is present and contains SOFA files in `male/` and `female/` subdirectories.

### Step 2: Configure Paths (If Needed)

If your directory structure differs from the default, edit the paths at the top of `experiment.py`:

```python
HRTF_DIR = r"C:/Users/Apermesa/Documents/Raytraced Audio/HRTF Dataset"
OUT_DIR  = r"C:/Users/Apermesa/Documents/Raytraced Audio/figures"
```

Change these to match your local directory paths.

### Step 3: Run the Experiment

```bash
python experiment.py
```

### Expected Output

The script will:

1. **Load HRTF data** -- reads 20 SOFA files (10 male, 10 female)
2. **Apply 5 degradation conditions:**
   - Minimum-phase reconstruction
   - 1/3-octave magnitude smoothing
   - 1/1-octave magnitude smoothing
   - 12-bit quantization
   - 8-bit quantization
3. **Compute 4 metrics** for each condition across all subjects:
   - Log-Spectral Distortion (LSD)
   - Interaural Level Difference Error (ILD)
   - Group Delay Error (GD)
   - Interaural Time Difference Error (ITD)
4. **Compute the Composite Perceptual Metric (CPM)** -- weighted combination
5. **Generate 4 figures** saved to `figures/`:
   - `fig4_1_metric_comparison.png` -- Grouped bar chart of normalized metrics
   - `fig4_2_lsd_vs_cpm.png` -- Scatter plot showing LSD vs CPM divergence
   - `fig4_3_spectral_comparison.png` -- Spectral comparison (original vs min-phase)
   - `fig4_4_group_delay.png` -- Group delay comparison revealing phase distortion
6. **Save numerical results** to `figures/results.csv`

### Expected Runtime

- Approximately 5-15 minutes depending on hardware (processes 20 subjects x 865 spatial positions x 5 conditions)

### Expected Results Summary

| Condition        | LSD (dB) | ILD Err (dB) | GD Err (ms) | ITD Err (ms) | CPM    |
|-----------------|-----------|---------------|-------------|--------------|--------|
| Min-Phase       | 0.030     | 0.058         | 2.335       | 0.579        | 14.094 |
| 1/3-Oct Smooth  | 2.332     | 3.122         | 0.176       | 0.014        | 1.880  |
| 1/1-Oct Smooth  | 3.829     | 4.562         | 0.209       | 0.023        | 2.592  |
| 12-bit Quant.   | 0.140     | 0.238         | 0.060       | 0.003        | 0.352  |
| 8-bit Quant.    | 1.065     | 1.730         | 0.202       | 0.012        | 1.503  |

---

## Key Results Interpretation

The **"LSD Blindspot"** is the central finding:

- The minimum-phase condition has the **lowest LSD** (0.030 dB) -- LSD says it is the "best" processing
- Yet it has the **highest CPM** (14.094) -- the composite metric says it is the "worst"
- The driver is the **group delay error** (2.335 ms), which is 10x larger than any other condition
- This confirms that magnitude-only metrics like LSD are blind to phase distortion

For the other four conditions, both LSD and CPM increase monotonically with degradation severity, confirming the CPM does not sacrifice magnitude sensitivity for phase sensitivity.

---

## CPM Formula

```
CPM = (0.20 * LSD / 2.0) + (0.25 * ILD_err / 1.0) + (0.35 * GD_err / 0.08) + (0.20 * ITD_err / 0.03)
```

Where normalization references are approximate JND (Just Noticeable Difference) thresholds.

---

## License

This project is for academic use. The RIEC HRTF Database is subject to its own usage terms as specified by Tohoku University.

---

## References

- Andreopoulou, A. & Katz, B. F. G. (2022). Perceptual Impact on Localization Quality Evaluations of Common Pre-Processing for Non-Individual HRTFs. *J. Audio Eng. Soc.*
- Watanabe, K. et al. (2014). Dataset of head-related transfer functions measured with a circular loudspeaker array. *Acoustical Science and Technology*, 35(3).
- Full bibliography available in `references.bib`
