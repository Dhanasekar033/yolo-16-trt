#!/usr/bin/env bash
# Build the Label Inspector with PyInstaller and install it as a desktop app.
#
#   ./install.sh              build from run.spec, then install
#   ./install.sh --no-build   install whatever is already in dist/run
#
# What ends up where:
#
#   /opt/label-inspector/                     the binary and its _internal payload
#   ~/.local/share/applications/…desktop      the entry in the applications menu
#   ~/Desktop/…desktop                        the icon that is double-clicked
#
# The install directory is handed to the operator's own account rather than
# left owned by root. The app keeps config.json, result/ and labels/ beside
# the executable -- app_dir() in utils/config.py -- and writes config.json on
# first run, so a root-owned /opt would leave it unable to save its own
# settings or record a run.
#
# Override any of these from the environment:
#   INSTALL_DIR   where the application goes      (default /opt/label-inspector)
#   DESKTOP_DIR   where the double-click icon goes
#   APPS_DIR      where the menu entry goes

set -euo pipefail

APP_NAME="label-inspector"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INSTALL_DIR="${INSTALL_DIR:-/opt/$APP_NAME}"
DESKTOP_DIR="${DESKTOP_DIR:-$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")}"
APPS_DIR="${APPS_DIR:-$HOME/.local/share/applications}"

SPEC="$SRC_DIR/run.spec"
ICON_SRC="$SRC_DIR/v_updated_logo.png"
ICON_NAME="$(basename "$ICON_SRC")"
BUILT="$SRC_DIR/dist/run"
ENTRY="$APP_NAME.desktop"

BUILD=1
case "${1:-}" in
    --no-build) BUILD=0 ;;
    "")         ;;
    *)          echo "usage: $0 [--no-build]" >&2; exit 2 ;;
esac

say() { printf '\n\033[1;32m==>\033[0m %s\n' "$1"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

# ── preflight ────────────────────────────────────────────────────────────
[ -f "$SPEC" ]     || die "no run.spec beside this script ($SPEC)"
[ -f "$ICON_SRC" ] || die "icon not found: $ICON_SRC"
python3 -c 'import PyInstaller' 2>/dev/null \
    || die "PyInstaller is not installed — python3 -m pip install --user pyinstaller"

# ── 1. build ─────────────────────────────────────────────────────────────
if [ "$BUILD" -eq 1 ]; then
    say "Building from run.spec — this takes a few minutes"
    cd "$SRC_DIR"
    python3 -m PyInstaller --noconfirm run.spec
else
    say "Skipping the build (--no-build)"
fi
[ -x "$BUILT/run" ] || die "no binary at $BUILT/run — run again without --no-build"

# ── 2. install ───────────────────────────────────────────────────────────
# Try without root first, so a non-system INSTALL_DIR needs no password.
SUDO=""
if ! mkdir -p "$INSTALL_DIR" 2>/dev/null; then
    SUDO="sudo"
    say "$INSTALL_DIR needs root — you will be asked for your password"
    $SUDO mkdir -p "$INSTALL_DIR"
fi

say "Installing into $INSTALL_DIR"
# --delete clears out what a previous build left behind, but config.json and
# everything the app records belong to the machine, not to the build: they
# are excluded so an upgrade never wipes settings someone filled in on site.
$SUDO rsync -a --delete \
    --exclude "config.json" \
    --exclude "result/" \
    --exclude "labels/" \
    "$BUILT"/ "$INSTALL_DIR"/
$SUDO cp "$ICON_SRC" "$INSTALL_DIR/$ICON_NAME"
$SUDO chown -R "$(id -un):$(id -gn)" "$INSTALL_DIR"
$SUDO chmod 644 "$INSTALL_DIR/$ICON_NAME"

# ── 3. the desktop entry ─────────────────────────────────────────────────
say "Creating the desktop icon"
mkdir -p "$APPS_DIR" "$DESKTOP_DIR"

# env RESOURCE_NAME=… is read by Qt's xcb plugin for the window's WM_CLASS.
# The app builds its QApplication with an empty argv, so without this the
# window would come up as "run" and the taskbar would not match it to this
# entry's icon.
cat > "$APPS_DIR/$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Label Inspector
GenericName=QR Label Validation Console
Comment=Camera + TensorRT rolling-window QR label validation
Exec=env RESOURCE_NAME=$APP_NAME $INSTALL_DIR/run
Path=$INSTALL_DIR
Icon=$INSTALL_DIR/$ICON_NAME
Terminal=false
StartupNotify=true
StartupWMClass=$APP_NAME
Categories=Utility;
Keywords=QR;label;inspection;vikbot;
Actions=Console;

[Desktop Action Console]
Name=Open with visible console
Exec=gnome-terminal --title=Label Inspector --working-directory=$INSTALL_DIR -- $INSTALL_DIR/run
EOF

command -v desktop-file-validate >/dev/null \
    && desktop-file-validate "$APPS_DIR/$ENTRY"

cp "$APPS_DIR/$ENTRY" "$DESKTOP_DIR/$ENTRY"
# GNOME refuses to launch a .desktop sitting on the Desktop unless it is both
# executable and flagged trusted. Without the pair it shows up as "Untrusted
# application launcher" and a double-click does nothing at all.
chmod +x "$DESKTOP_DIR/$ENTRY"
gio set "$DESKTOP_DIR/$ENTRY" metadata::trusted true 2>/dev/null || true
command -v update-desktop-database >/dev/null \
    && update-desktop-database "$APPS_DIR" 2>/dev/null || true

# ── done ─────────────────────────────────────────────────────────────────
say "Installed"
cat <<EOF

  application   $INSTALL_DIR/run
  icon          $DESKTOP_DIR/$ENTRY
  menu entry    $APPS_DIR/$ENTRY

Double-click "Label Inspector" on the desktop to start it. Right-click it for
"Open with visible console" when you need to see what it prints.

config.json is written beside the application on first run — edit
$INSTALL_DIR/config.json to set the camera, the relay and the thresholds for
this machine. Reinstalling leaves that file, result/ and labels/ alone.
EOF
