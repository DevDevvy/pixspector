from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pixspector.cli import app

runner = CliRunner()


def _make_temp_img(tmp_path: Path) -> Path:
    import numpy as np
    import cv2
    p = tmp_path / "temp.jpg"
    # simple gradient with a square (JPEG)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img[:, i, :] = i
    cv2.rectangle(img, (64, 64), (192, 192), (255, 255, 255), -1)
    cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return p


def test_version():
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "pixspector" in res.stdout


def test_analyze_single(tmp_path: Path):
    img = _make_temp_img(tmp_path)
    out = tmp_path / "out"
    res = runner.invoke(app, ["analyze", str(img), "--report", str(out), "--no-pdf"])
    assert res.exit_code == 0
    # check artifacts
    json_path = out / f"{img.stem}_report.json"
    assert json_path.exists()
    # summary table mentions JSON file name
    assert f"{img.stem}_report.json" in res.stdout
