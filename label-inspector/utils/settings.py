"""What the console remembers between runs.

Two things the operator sets from the screen have to survive a restart, or
they have to be set again every morning: the folder the label crops are
written to, and which sheets have been loaded before. Everything else about a
run is either on the command line or in the record itself.

It lives outside the project, next to the voice cache:

    ~/.config/label-inspector/settings.json

so a `git pull` over the project, or a copy of it to another machine, cannot
take the machine's own configuration with it.

Nothing here is allowed to stop the line. A settings file that is missing,
unreadable, corrupt or on a full disk costs the operator a remembered folder
and nothing more, so every path through this module ends in a working object.
"""

import json
import os

# LABEL_INSPECTOR_SETTINGS moves the file, which is how a second install on
# the same machine -- or a test run -- keeps its hands off the operator's.
DEFAULT_PATH = os.environ.get("LABEL_INSPECTOR_SETTINGS") or \
    os.path.expanduser("~/.config/label-inspector/settings.json")
KEEP_RECENT = 8          # sheets remembered; the button is a menu, not a list


class Settings:
    """The remembered folder and the recently loaded sheets.

    Reads on construction and writes on every change -- there are only a
    handful of changes in a shift, all of them made by a human clicking a
    button, so there is nothing to batch up.
    """

    def __init__(self, path=DEFAULT_PATH):
        self.path = path
        self.label_dir = None
        # Where CAPTURE FRAME writes. A folder of its own, remembered like
        # the crops folder and for the same reason: it is a place on this
        # machine's disk, chosen at the machine, and a USB stick that was
        # plugged in on Monday should still be the folder on Tuesday.
        self.capture_dir = None
        # Exposure, gain and brightness, as the operator left them. They
        # belong here rather than in config.json: config.json is how an
        # installation is set up, this is what somebody adjusted at the
        # machine, and the two should not be able to overwrite each other.
        self.camera = {}
        # Which way the codes are being read, and the four margins the label
        # crop is cut by when they are read straight off the picture. Set at
        # the machine, against the live picture, so they belong here rather
        # than in config.json for the same reason the camera does.
        self.scan = {}
        # Which ups across the web are being checked, as up numbers -- UP1
        # is the first label across -- or None for all of them. It is the
        # reel that decides this, not the sheet: a four-up sheet run three
        # up would stop the line at every row for the label that is not
        # there, and the operator would have to tick that up off again
        # every morning.
        self.ups = None
        # Which sheet column each up is checked against, as column numbers in
        # up order. None is straight through, which is what a fresh install
        # starts on and what nearly every sheet wants.
        self.ups_map = None
        # MJPG or YUYV -- which the console was last left on. A property of
        # how this reel is being checked, not of a session: the operator who
        # chose uncompressed frames for a fine print job should not have to
        # choose again in the morning.
        self.format = None
        self._recent = []
        self._warned = False
        self._load()

    # -- reading ----------------------------------------------------------
    def _load(self):
        try:
            with open(self.path) as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            print(f"[settings] ignoring {self.path} ({exc})")
            return
        if not isinstance(data, dict):
            return
        folder = data.get("label_dir")
        if isinstance(folder, str) and folder:
            self.label_dir = folder
        folder = data.get("capture_dir")
        if isinstance(folder, str) and folder:
            self.capture_dir = folder
        recent = data.get("recent")
        if isinstance(recent, list):
            self._recent = [p for p in recent if isinstance(p, str) and p]
        camera = data.get("camera")
        if isinstance(camera, dict):
            self.camera = {k: int(v) for k, v in camera.items()
                           if isinstance(v, (int, float))}
        scan = data.get("scan")
        if isinstance(scan, dict):
            self.scan = scan
        fmt = data.get("format")
        if isinstance(fmt, str) and fmt.upper() in ("MJPG", "YUYV"):
            self.format = fmt.upper()
        ups_map = data.get("ups_map")
        if isinstance(ups_map, list):
            cols = [int(n) for n in ups_map
                    if isinstance(n, (int, float)) and int(n) >= 1]
            self.ups_map = cols if len(set(cols)) == len(cols) and cols else None
        ups = data.get("ups")
        if isinstance(ups, list):
            wanted = sorted({int(n) for n in ups
                             if isinstance(n, (int, float)) and n >= 1})
            self.ups = wanted or None

    @property
    def recent(self):
        """Sheets loaded before, newest first, that are still on disk.

        Filtered on the way out rather than on the way in: a sheet on a USB
        stick that is unplugged today is still the sheet that was run
        yesterday, and comes back on the menu when the stick does.
        """
        return [p for p in self._recent if os.path.exists(p)]

    @property
    def sheet(self):
        """The last sheet loaded, if it is still there."""
        return next(iter(self.recent), None)

    # -- writing ----------------------------------------------------------
    def remember_sheet(self, path):
        if not path:
            return
        path = os.path.abspath(path)
        self._recent = [path] + [p for p in self._recent
                                 if p != path][:KEEP_RECENT - 1]
        self._save()

    def remember_label_dir(self, path):
        if not path:
            return
        self.label_dir = os.path.abspath(path)
        self._save()

    def remember_capture_dir(self, path):
        if not path:
            return
        self.capture_dir = os.path.abspath(path)
        self._save()

    def remember_camera(self, values):
        """Where the exposure, gain and brightness sliders were left.

        Called on every slider move, which is more often than anything else
        here writes -- but that is a handful of writes while somebody drags
        a slider, not a rate, and losing the setting because the console was
        switched off at the wall is the thing worth avoiding.
        """
        if not values:
            return
        merged = dict(self.camera)
        merged.update({k: int(v) for k, v in values.items() if v is not None})
        if merged == self.camera:
            return
        self.camera = merged
        self._save()

    def remember_scan(self, direct=None, pad=None, scale=None):
        """How the codes are being read, as the operator left it.

        Written on every change, like the camera and for the same reason:
        the padding is set by watching the picture and then walking back to
        the machine, and a console switched off at the wall must not lose
        what was just dialled in.
        """
        scan = dict(self.scan)
        if direct is not None:
            scan["direct"] = bool(direct)
        if pad:
            scan["pad"] = {k: int(v) for k, v in pad.items()}
        if scale is not None:
            scan["scale"] = float(scale)
        if scan == self.scan:
            return
        self.scan = scan
        self._save()

    def remember_ups(self, numbers):
        """Which ups the tick boxes were left on, as up numbers.

        None -- or every up the sheet has -- is written as null, which is
        what a fresh install starts on: check whatever the sheet asks for.
        """
        wanted = sorted({int(n) for n in numbers}) if numbers else None
        if wanted == self.ups:
            return
        self.ups = wanted
        self._save()

    def remember_format(self, name):
        """Which camera format the console was last left on."""
        name = (name or "").upper()
        if name not in ("MJPG", "YUYV") or name == self.format:
            return
        self.format = name
        self._save()

    def remember_ups_map(self, numbers):
        """Which column each up was left pointing at, as column numbers.

        Straight through is written as null: it is the default, and a file
        that spells it out would freeze today's number of ups into a setting
        that is meant to follow the sheet.
        """
        wanted = [int(n) for n in numbers] if numbers else None
        if wanted and wanted == list(range(1, len(wanted) + 1)):
            wanted = None
        if wanted == self.ups_map:
            return
        self.ups_map = wanted
        self._save()

    def _save(self):
        data = {"label_dir": self.label_dir,
                "capture_dir": self.capture_dir, "recent": self._recent,
                "camera": self.camera, "scan": self.scan, "ups": self.ups,
                "ups_map": self.ups_map, "format": self.format}
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, self.path)      # never a half-written settings file
        except OSError as exc:
            if not self._warned:            # once, not once per click
                self._warned = True
                print(f"[settings] cannot save to {self.path} ({exc}); this "
                      f"session's choices will not be remembered")
