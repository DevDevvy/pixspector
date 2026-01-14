# Architecture

## Text-form architecture diagram

```
[1] Image input (CLI/GUI)
        |
        v
[2] Load & normalize (resize, color channels, SHA256)
        |
        v
[3] Metadata & provenance checks
    - EXIF/XMP/IPTC extraction
    - Optional C2PA verification
        |
        v
[4] Concurrent forensic analyses
    - ELA, JPEG ghosts, DCT/Benford
    - Resampling, CFA, PRNU, FFT
    - AI signal heuristics
        |
        v
[5] Provenance flags & module aggregation
        |
        v
[6] Rule-based scoring (Suspicion Index)
        |
        v
[7] Report assembly (JSON model + visuals)
        |
        v
[8] Outputs
    - JSON report
    - PDF report
    - PNG artifacts
```

## Pipeline stages (1–8)

1. **Image input (CLI/GUI)**
   - **Purpose:** Accept user-selected images and initialize a run.
   - **Tools/APIs:** CLI entry point (`pixspector analyze`) or GUI app.
   - **Outputs:** Image path(s) and output directory selection.

2. **Load & normalize**
   - **Purpose:** Load the image, normalize size and color spaces for consistent analysis.
   - **Tools/APIs:** `load_image`, `to_float32`, `ensure_color` utilities.
   - **Outputs:** BGR/RGB/gray arrays, dimensions, scale factor, SHA256, input artifact.

3. **Metadata & provenance checks**
   - **Purpose:** Extract metadata and verify provenance claims.
   - **Tools/APIs:** `read_metadata`; optional `c2patool` via `c2pa.verify`.
   - **Outputs:** EXIF/XMP/IPTC fields, JPEG tables/ICC data, C2PA validity signals.

4. **Concurrent forensic analyses**
   - **Purpose:** Run core forensic modules in parallel for speed and coverage.
   - **Tools/APIs:** `run_ela`, `run_jpeg_ghosts`, `run_dct_benford`,
     `run_resampling_map`, `run_cfa_map`, `run_prnu`, `run_fft_checks`,
     `run_ai_detection`; `ThreadPoolExecutor`.
   - **Outputs:** Module metrics plus visual artifacts (diff maps, curves, overlays).

5. **Provenance flags & module aggregation**
   - **Purpose:** Consolidate results and derive provenance signals used by scoring.
   - **Tools/APIs:** `_provenance_flags` helper, module result collection.
   - **Outputs:** `modules` dictionary with provenance flags and analysis results.

6. **Rule-based scoring (Suspicion Index)**
   - **Purpose:** Convert evidence into a transparent, weighted score and bucket.
   - **Tools/APIs:** `score_image` with configurable weights/buckets.
   - **Outputs:** Suspicion Index (0–100), bucket label, evidence list, notes.

7. **Report assembly (JSON model + visuals)**
   - **Purpose:** Build the structured report payload and connect it to artifacts.
   - **Tools/APIs:** Report dictionary assembly, `_convert_to_serializable`.
   - **Outputs:** Report object with metadata, module outputs, scoring, artifact paths.

8. **Outputs (JSON/PDF/PNG)**
   - **Purpose:** Persist results for downstream consumption.
   - **Tools/APIs:** `save_json_report`, `save_pdf_report`, `save_image_png` helpers.
   - **Outputs:** `*_report.json`, optional `*_report.pdf`, and PNG artifacts folder.
