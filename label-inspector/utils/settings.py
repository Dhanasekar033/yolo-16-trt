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
        # Exposure, gain and brightness, as the operator left them. They
        # belong here rather than in config.json: config.json is how an
        # installation is set up, this is what somebody adjusted at the
        # machine, and the two should not be able to overwrite each other.
        self.camera = {}
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
        recent = data.get("recent")
        if isinstance(recent, list):
            self._recent = [p for p in recent if isinstance(p, str) and p]
        camera = data.get("camera")
        if isinstance(camera, dict):
            self.camera = {k: int(v) for k, v in camera.items()
                           if isinstance(v, (int, float))}

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

    def _save(self):
        data = {"label_dir": self.label_dir, "recent": self._recent,
                "camera": self.camera}
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
