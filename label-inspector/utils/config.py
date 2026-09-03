"""The settings that belong to an installation rather than to the code.

This app is meant to be handed over as one bundled application, put on a
panel PC next to a winder, and configured there -- by somebody who has the
machine in front of them and not the source. Which camera, how big the
picture is, which relay starts the motor, how long the read-in runs: all of
that is a property of one installation, and none of it should need a rebuild
to change.

So it lives in config.json, beside the application, and this module is what
finds it. Every value in it is a default -- the command line still wins, so
a run can be told something different for one shift without editing the
file, and a value left out of the file falls back to what is written here.
A file that is missing is written out fresh on first run, filled in with
these defaults, which makes it self-documenting: install, run once, open
config.json and everything that can be set is already in it with its
current value.

Frozen or from source, two directories matter and they are not the same one:

    bundle_dir()   where read-only things that ship with the app are -- the
                   .engine, classes.txt. Under PyInstaller this is the
                   unpacked temporary directory, which is gone at exit.
    app_dir()      where the application itself sits, which is where
                   config.json is read from and where anything the app
                   writes has to go. Writing to bundle_dir() under
                   PyInstaller means writing to a folder that gets deleted.

Run from source the two are the same folder, which is why the difference is
easy to miss until the day it is packaged.
"""

import copy
import json
import os
import sys

# What an installation gets before anyone changes anything. Also the whole
# vocabulary of config.json: a key that is not here is one this app does not
# read, and it says so on load rather than being quietly ignored.
DEFAULTS = {
    "camera": {
        # Found by name across /dev/videoN, so a device that moves between
        # boots is still the right camera. `index` pins it if that fails.
        "name": "Global Shutter Camera",
        "index": None,
        "device": None,          # for the exposure/gain/brightness controls;
                                 # null -> /dev/video<the index in use>
        "width": 2592,
        "height": 1944,
        "fps": 60,               # MJPG does 60 at full size; YUYV only 35
        "format": "MJPG",
        "rotate": 270,           # 0/90/180/270, done inside the pipeline
        # How far the sliders behind 's' are allowed to travel, narrowing
        # what the camera itself reports. This one goes to 10000, but a
        # frame that takes longer than the web takes to move a label is a
        # smeared frame, so the top of that range is exposure settings that
        # can only make the reading worse -- and a slider that spends nine
        # tenths of its length in useless territory is a slider nobody can
        # set. [min, max], or null for whatever the device allows.
        "limits": {
            "exposure": [1, 1000],
            "gain": None,
            "brightness": None,
        },
    },
    "display": {
        "max_width": 1280,       # the window is capped to this, so a full
        "max_height": 960,       # 5MP frame does not overflow the screen
    },
    "model": {
        "engine": "best-new.engine",
        "classes": "classes.txt",
        "imgsz": 640,
        "conf": 0.45,            # for any class without its own
        "conf_label": None,
        "conf_qr": None,
        "label_class": "label",
        "qr_class": "qr_code",
        "logo_class": "logo",
    },
    "decode": {
        "qr_margin": 0.15,       # quiet zone round the code box
        "qr_margin_min": 8,
        "zbar_fallback": 2,      # labels per frame that may fall back to zbar
    },
    "machine": {
        "start_delay": 2.0,      # seconds of reading before the relay goes on
        "window_size": 8,
        "hold_after": 2,
        "part_looks": 8,
        "no_read_secs": 1.5,
        "rewind_clear": 2.5,
        "dm_repeats": 3,         # printings of each datamatrix down the web
        "max_ups": 8,            # ups the voice is warmed up for before a
                                 # sheet says how many there really are
    },
    "relay": {
        "port": None,            # null -> auto-detect
        "start": 0,              # the motor, and the only one this app drives
    },
    "paths": {
        # Relative paths are taken as relative to app_dir(), never to
        # whatever directory the app happened to be launched from -- a
        # desktop shortcut can be launched from anywhere.
        "result_dir": "result",
        "label_dir": None,       # null -> <app_dir>/labels, and the console
                                 # remembers whatever the operator picks
    },
    "voice": {
        "engine": "auto",        # auto | edge | espeak
        "name": None,            # female | male | expressive | a voice name
        "rate": 10,              # percent off normal
    },
}

CONFIG_NAME = "config.json"


def frozen():
    """Is this the bundled application rather than the source?"""
    return getattr(sys, "frozen", False)


def bundle_dir():
    """Where the things that ship with the app are: the engine, classes.txt."""
    if frozen():
        # One-file builds unpack here; one-folder builds set it too.
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def app_dir():
    """Where the application sits, and where anything it writes belongs."""
    if frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path():
    # LABEL_INSPECTOR_CONFIG points the whole thing somewhere else, which is
    # how a second install on one machine keeps its own.
    return os.environ.get("LABEL_INSPECTOR_CONFIG") or \
        os.path.join(app_dir(), CONFIG_NAME)


def _merge(base, over, path=""):
    """Overlay a loaded file on the defaults, one leaf at a time.

    Section by section rather than wholesale, so a file that sets one value
    in a section keeps the defaults for the rest of it -- and a key nobody
    recognises is said out loud, because a silently ignored setting is worse
    than no setting at all.
    """
    for key, value in (over or {}).items():
        where = f"{path}{key}"
        if key not in base:
            print(f"[config] {config_path()}: '{where}' is not a setting this "
                  f"app reads — ignored")
            continue
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value, f"{where}.")
        else:
            base[key] = value
    return base


class Config(dict):
    """config.json, with the defaults filled in behind it."""

    def __init__(self, path=None, write_missing=True, quiet=False):
        super().__init__(copy.deepcopy(DEFAULTS))
        self.path = path or config_path()
        self.loaded = False
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            if write_missing:
                self.save(quiet=quiet)
            return
        except (OSError, ValueError) as exc:
            # A broken config must not stop a line. Say so loudly and run on
            # the defaults, which are the values this was shipped with.
            print(f"[config] {self.path} could not be read ({exc}) — running "
                  f"on the built-in defaults")
            return
        if not isinstance(data, dict):
            print(f"[config] {self.path} is not a JSON object — running on "
                  f"the built-in defaults")
            return
        _merge(self, data)
        self.loaded = True
        if not quiet:
            print(f"[config] read {self.path}")

    # ── reading ──────────────────────────────────────────────────────────
    def get_path(self, value, root=None):
        """A configured path, made absolute against the application.

        Relative paths in config.json mean 'beside the application', because
        the alternative -- relative to the working directory -- means a
        desktop shortcut writes its record wherever the desktop happens to
        be.
        """
        if not value:
            return None
        return value if os.path.isabs(value) else \
            os.path.join(root or app_dir(), value)

    def asset(self, value):
        """A file that ships with the app: looked for beside the application
        first, so an operator can drop in a new engine without a rebuild,
        then inside the bundle."""
        if not value:
            return None
        if os.path.isabs(value):
            return value
        beside = os.path.join(app_dir(), value)
        if os.path.exists(beside):
            return beside
        return os.path.join(bundle_dir(), value)

    # ── writing ──────────────────────────────────────────────────────────
    def save(self, quiet=False):
        """Write the file out, so what can be set is visible and editable."""
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(self, fh, indent=2)
                fh.write("\n")
            os.replace(tmp, self.path)      # never a half-written config
            if not quiet:
                print(f"[config] wrote {self.path}")
            return True
        except OSError as exc:
            print(f"[config] cannot write {self.path} ({exc}); this run uses "
                  f"the built-in defaults")
            return False
