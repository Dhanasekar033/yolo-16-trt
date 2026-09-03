# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

# cuda-python is Cython: cuda/bindings/driver.pyx cimports cydriver, and a
# cimport leaves nothing in the bytecode for PyInstaller's import scanner to
# follow. So none of the compiled cuda.bindings.* extensions are collected on
# their own, and the app dies at trt_engine.py's `from cuda.bindings import
# runtime`. collect_all takes the whole package -- the .so submodules and the
# data cuda.pathfinder needs to find the CUDA libraries at run time.
#
# tensorrt is deliberately not listed: pyinstaller-hooks-contrib already ships
# a hook for it, which is why `import tensorrt` was the one line in
# trt_engine.py that did not fail.
cuda_datas, cuda_binaries, cuda_hiddenimports = collect_all('cuda')


# What ships with the application, and what it is given at the machine, are
# not the same list.
#
#   datas          things the app cannot run without and that never change
#                  per installation: the engine and its class names. These
#                  are found through Config.asset(), which looks beside the
#                  application first -- so a new engine can be dropped in
#                  next to the exe without a rebuild -- and falls back to
#                  the bundled copy.
#
#   NOT bundled    config.json, and the record and crop folders. config.json
#                  is written on first run beside the executable, filled in
#                  with the defaults, for whoever installs it to edit. It
#                  must not be baked in: under a one-file build the bundle
#                  is unpacked to a temp directory that is deleted on exit,
#                  so a config.json inside it could never be edited and
#                  anything written next to it would be thrown away. This is
#                  why utils/config.py separates bundle_dir() from app_dir().

a = Analysis(
    ['run.py'],
    pathex=[],
    binaries=cuda_binaries,
    datas=[
        ('best-new.engine', '.'),
        ('classes.txt', '.'),
        ('vikbot-logo.png', '.'),
    ] + cuda_datas,
    hiddenimports=cuda_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='run',
)
