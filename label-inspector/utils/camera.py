"""Exposure, gain and brightness, straight to the camera.

The picture the model gets is only as good as the light in it, and on this
line that is not a thing set once: a reel with more gloss, a lamp that has
aged, a different shift's ambient light, and codes that read all morning
stop reading. Those three controls are the fix, and they belong on the
console with the operator rather than in a separate tool.

They are set the way V4L2 sets anything -- an ioctl on the device node --
and not through OpenCV, whose exposure and gain properties do nothing at all
under the GStreamer backend this app captures with. Nor through v4l2-ctl,
which would be one more thing to install on a machine meant to run a single
bundled application. It is ioctl and the standard library, so it works the
same frozen as it does from source, on a machine with nothing else on it.

A second file descriptor on a device that GStreamer is already streaming
from is fine: V4L2 makes streaming exclusive, not the controls.

Nothing here is allowed to stop the line. A camera that will not open, a
control the device does not have, a value it refuses -- each costs the
operator a slider and nothing else.
"""

import fcntl
import os
import re
import shutil
import struct
import subprocess

# ── the ioctls ───────────────────────────────────────────────────────────
def _IOWR(kind, nr, size):
    return (3 << 30) | (size << 16) | (ord(kind) << 8) | nr


VIDIOC_QUERYCTRL = _IOWR("V", 36, 68)     # struct v4l2_queryctrl
VIDIOC_G_CTRL    = _IOWR("V", 27, 8)      # struct v4l2_control
VIDIOC_S_CTRL    = _IOWR("V", 28, 8)

_QUERYCTRL = "II32siiiiIII"               # id, type, name[32], min, max,
_CTRL      = "Ii"                         # step, default, flags, reserved[2]

V4L2_CTRL_FLAG_DISABLED = 0x0001

# What the console offers, in the order it offers it. The ids are V4L2's own.
EXPOSURE   = "exposure"
GAIN       = "gain"
BRIGHTNESS = "brightness"

CONTROLS = {
    EXPOSURE:   (0x009A0902, "Exposure"),
    GAIN:       (0x00980913, "Gain"),
    BRIGHTNESS: (0x00980900, "Brightness"),
}

EXPOSURE_AUTO = 0x009A0901       # the menu that has to say manual first
EXPOSURE_MANUAL = 1              # V4L2_EXPOSURE_MANUAL


def list_rates(device, width, height):
    """{"MJPG": [rates], "YUYV": [rates]} at one size, straight from the driver.

    What a format is worth is not a number anyone can write down once: MJPG
    is compressed in the camera and gets a high rate; YUYV is the sensor's
    own pixels and is limited by what the USB link will carry, which depends
    on the size, the camera and the cable. This rig's 5MP camera did 60 and
    35; the 2MP one on it now does 90 and 5 at full frame. So the console
    asks rather than tells, and a format nobody can use at this size shows
    the rate that says so.
    """
    if not shutil.which("v4l2-ctl"):
        return {}
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}

    rates, fmt, size = {}, None, None
    for line in out.splitlines():
        hit = re.search(r"\]:\s*'(\w+)'", line)
        if hit:
            fmt = hit.group(1).upper()
            continue
        hit = re.search(r"Size:\s*\w+\s*(\d+)x(\d+)", line)
        if hit:
            size = (int(hit.group(1)), int(hit.group(2)))
            continue
        hit = re.search(r"\(([\d.]+)\s*fps\)", line)
        if hit and fmt and size == (int(width), int(height)):
            rates.setdefault(fmt, []).append(float(hit.group(1)))
    return {k: sorted(v, reverse=True) for k, v in rates.items()}


class CameraControls:
    """The three controls on one camera: what they allow, and what they are.

    Opened lazily and reopened on demand, so a camera unplugged and put back
    does not leave the console holding a dead descriptor.
    """

    def __init__(self, device="/dev/video0", limits=None, enabled=True):
        self.device = device
        # Off when the frames are coming from a recording: there is no device
        # behind them, and probing one would print a warning about a camera
        # nobody asked for.
        self.enabled = enabled
        # What the console may offer, narrower than what the device allows.
        # A camera's full exposure range runs far past anything usable on a
        # moving web -- see config.json's camera.limits.
        self.limits = {k: v for k, v in (limits or {}).items() if v}
        self.ranges = {}         # name -> {min, max, step, default}
        self._fd = None
        self._warned = False
        if enabled:
            self.probe()

    # ── the device ───────────────────────────────────────────────────────
    def _open(self):
        if not self.enabled:
            return None
        if self._fd is not None:
            return self._fd
        try:
            self._fd = os.open(self.device, os.O_RDWR)
        except OSError as exc:
            if not self._warned:
                self._warned = True
                print(f"[camera] cannot open {self.device} for controls "
                      f"({exc}); exposure, gain and brightness are not "
                      f"adjustable from the console")
            self._fd = None
        return self._fd

    def close(self):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def _drop(self):
        """Let go of a descriptor that has stopped answering."""
        self.close()

    # ── one control ──────────────────────────────────────────────────────
    def _query(self, cid):
        fd = self._open()
        if fd is None:
            return None
        buf = bytearray(struct.pack(_QUERYCTRL, cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf, True)
        except OSError:
            self._drop()
            return None
        _id, _type, _name, lo, hi, step, dflt, flags, _r0, _r1 = \
            struct.unpack(_QUERYCTRL, buf)
        if flags & V4L2_CTRL_FLAG_DISABLED or hi <= lo:
            return None
        return {"min": lo, "max": hi, "step": max(step, 1), "default": dflt}

    def probe(self):
        """What this camera actually offers, of the three, narrowed to what
        the installation allows. Ones it does not have simply do not appear,
        and the console shows no slider."""
        self.ranges = {}
        for name, (cid, _label) in CONTROLS.items():
            spec = self._query(cid)
            if spec is None:
                continue
            limit = self.limits.get(name)
            if limit:
                # Intersected, never widened: a configured limit can only
                # narrow what the hardware said, because the hardware is the
                # one that knows what it will accept.
                try:
                    lo, hi = int(limit[0]), int(limit[1])
                except (TypeError, ValueError, IndexError):
                    lo = hi = None
                if lo is not None and hi > lo:
                    spec = dict(spec)
                    spec["min"] = max(spec["min"], lo)
                    spec["max"] = min(spec["max"], hi)
                    if spec["max"] <= spec["min"]:
                        # A limit that leaves nothing is a mistake in the
                        # config, not an instruction to remove the control.
                        print(f"[camera] config limit {limit} for {name} "
                              f"leaves no room inside the camera's own range "
                              f"— ignoring it")
                        spec = self._query(cid)
                    else:
                        spec["default"] = max(spec["min"],
                                              min(spec["default"], spec["max"]))
            self.ranges[name] = spec
        return self.ranges

    @property
    def available(self):
        return bool(self.ranges)

    def _get(self, cid):
        fd = self._open()
        if fd is None:
            return None
        buf = bytearray(struct.pack(_CTRL, cid, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_G_CTRL, buf, True)
        except OSError:
            self._drop()
            return None
        return struct.unpack(_CTRL, buf)[1]

    def _set(self, cid, value):
        fd = self._open()
        if fd is None:
            return False
        buf = bytearray(struct.pack(_CTRL, cid, int(value)))
        try:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, buf, True)
        except OSError:
            self._drop()
            return False
        return True

    # ── what the console talks to ────────────────────────────────────────
    def get(self, name):
        entry = CONTROLS.get(name)
        return None if entry is None else self._get(entry[0])

    def set(self, name, value):
        """Set one control, clamped to what the camera will take."""
        entry = CONTROLS.get(name)
        if entry is None or name not in self.ranges:
            return False
        spec = self.ranges[name]
        value = max(spec["min"], min(int(value), spec["max"]))
        if name == EXPOSURE:
            # A camera left on auto exposure quietly ignores every exposure
            # it is given, and the slider looks broken. Asking for a time is
            # asking for manual.
            self._set(EXPOSURE_AUTO, EXPOSURE_MANUAL)
        return self._set(entry[0], value)

    def snapshot(self):
        """Every control this camera has, as it is set right now."""
        out = {}
        for name in self.ranges:
            value = self.get(name)
            if value is not None:
                out[name] = value
        return out

    def apply(self, values):
        """Put a remembered set back. Returns the ones that took."""
        done = {}
        for name, value in (values or {}).items():
            if name in self.ranges and value is not None and self.set(name, value):
                done[name] = int(value)
        return done
