# PyInstaller spec for building the pixspector GUI as a single executable
# Usage:
#   pyinstaller pixspector_gui.spec

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules('skimage')

block_cipher = None

a = Analysis(
    ['-m', 'pixspector.gui.app'],
    pathex=[],
    binaries=[],
    datas=[
        ('config/defaults.yaml', 'config'),
    ],
    hiddenimports=hiddenimports + ['numpy', 'cv2', 'PIL', 'matplotlib', 'yaml', 'reportlab', 'jinja2', 'ExifRead', 'scipy', 'PySide6'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='pixspector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,       # set True if you want a terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
)
