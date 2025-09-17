# pixspector

**pixspector** is a **no-ML** (no deep learning) image-forensics toolkit you can run locally.
It provides explainable, classical analyses and a transparent rule-based **Suspicion Index (0–100)**.

## What it does

- **Provenance & metadata:** EXIF/XMP/IPTC, JPEG quantization tables, ICC profile; optional **C2PA** (Content Credentials) verification.
- **Classical forensic modules:**
  - **ELA** (Error Level Analysis)
  - **JPEG ghosts / double-JPEG** search & misalignment checks
  - **DCT/Benford** digit tests on AC coefficients
  - **Resampling detection** (scale/rotate/splice) via Radon/FFT periodicity
  - **CFA/demosaicing consistency** heatmap
  - **PRNU** (sensor noise residuals; optional camera gallery later)
  - **FFT checks** (2D FFT, radial spectrum, periodic peaks)
- **Scoring:** rule-based fusion → **Suspicion Index** with per-module evidence and a narrative explanation.
- **Reports:** exports **JSON** and **PDF** with all visualizations.

> ⚠️ This tool provides **evidence** and an **index**, not a legal verdict. Human review is essential.

## Quickstart

### 1) Using the bootstrap script

```bash
chmod +x scripts/bootstrap_structure.sh
./scripts/bootstrap_structure.sh
```
