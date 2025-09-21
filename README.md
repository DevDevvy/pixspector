# pixspector

**pixspector** is a classical, explainable **image forensics toolkit** — no machine learning involved.  
It provides evidence visualizations, rule-based scoring, and generates PDF/JSON reports, all offline.

## Features

- **Provenance & metadata**
  - Extracts EXIF/XMP/IPTC, JPEG quantization tables, ICC profile  
  - Optional **C2PA** (Content Credentials) verification

- **Forensic analyses**
  - **ELA** (Error Level Analysis)  
  - **JPEG ghosts / double-JPEG** search  
  - **DCT/Benford** law deviation  
  - **Resampling** detection (scale/rotate/splice)  
  - **CFA/demosaicing** consistency  
  - **PRNU residuals** (sensor noise fingerprinting)  
  - **FFT checks** (periodic peaks, high-frequency rolloff)  

- **Scoring**
  - Transparent, rule-based **Suspicion Index (0–100)**  
  - Evidence breakdown and reliability notes  

- **Reporting**
  - **JSON** with structured results  
  - **PDF** forensic report with visuals  

## Installation

### Quick setup
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a `.venv` virtual environment  
- Install all dependencies  
- Check for `c2patool` (optional, for C2PA checks)  

Activate the environment:
```bash
source .venv/bin/activate
```

Install pixspector into the environment so the CLI and GUI entry points are
available:
```bash
pip install -e .
```

## Usage

### CLI mode

Analyze a single image:
```bash
pixspector analyze examples/sample_images/edited_1.jpg --report out/
```

Analyze a folder:
```bash
pixspector analyze './photos/*.jpg' --report out/
```

Outputs:
- JSON report: `*_report.json`  
- PDF report: `*_report.pdf` (unless `--no-pdf` is used)  
- PNG visualizations in `out/<filename>_artifacts/`  

Check environment:
```bash
pixspector doctor
```

Show version:
```bash
pixspector version
```

### GUI (Desktop App)

You can run pixspector with a friendly GUI.

#### Dev mode
```bash
# in your activated venv
pixspector-gui
```

> **Tip:** If `pixspector-gui` is not recognized, ensure you have installed the
> package into the virtual environment (e.g., `pip install -e .`). You can also
> start the GUI directly with `python -m pixspector.gui.app`.

- Drag & drop images or click **Browse…**  
- Choose an output folder  
- Toggle **Generate PDF** (on/off)  
- Click **Analyze**  
- Preview visuals (ELA, resampling overlays, FFT, etc.)  
- Open the report folder with one click  

#### Build a standalone app

Using PyInstaller:
```bash
chmod +x scripts/build_app.sh
./scripts/build_app.sh
```

The built application will appear in `dist/pixspector/`.  
On Windows:
```bat
scripts\build_app_win.bat
```

## Example report

- **Suspicion Index:** 55 (Medium)  
- **Evidence:**
  - `ela_strong`: uneven recompression detected  
  - `jpeg_double_misaligned`: double-JPEG misalignment found  
- **Notes:**  
  C2PA manifest not found. Results are evidence, not proof.  

## License

MIT
