@echo off
setlocal enabledelayedexpansion
if not exist .venv (
  py -3 -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install pyinstaller

pyinstaller pixspector_gui.spec

echo Build complete. See dist\pixspector\
pause
