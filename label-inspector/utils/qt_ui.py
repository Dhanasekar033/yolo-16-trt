"""The operator console as a real window, instead of pixels drawn on the video.

The OpenCV console in ui.py has to rasterise every button, every caption and
every status line into the 1944x2592 frame sixty times a second, and that
frame is then shrunk to about a third for the screen with a nearest-neighbour
resize. Two things follow from that, and both are why this module exists.

It is slow. Measured on this machine at production frame size, the drawn
console plus `imshow` and `waitKey(1)` cost about 5.7ms of a 16.7ms frame
budget; the same screen through Qt costs about 0.6ms. `waitKey(1)` alone is
1.06ms of it, and OpenCV here is built against GTK2, which is the oldest and
slowest way there is to get a pixel onto a screen.

It looks bad. Nearest-neighbour downscaling of antialiased glyphs is exactly
what makes text chunky, so every caption had to be drawn oversized to survive
the shrink -- which is why the picture ended up covered in text. Here the
chrome is made of widgets and the overlay is painted at display resolution
with real font rendering, so the same information takes roughly a third of the
area and reads better.

The split of work: run.py owns the machine and decides *what* to show, and
hands over a snapshot of drawing instructions in display coordinates. This
module owns *how*, and knows nothing about sheets, windows or faults. It sends
back nothing but named commands.

Threading. Qt owns the main thread, so run.py's capture loop runs on a worker
and posts snapshots across with a queued signal. Nothing here touches the
machine's state directly: every button and key turns into a command that the
worker picks up at the top of its next pass, which is what keeps the state
single-threaded and keeps a 150ms relay write off the GUI thread.
"""

import html
import os
import re
import time

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

from utils.config import app_dir, bundle_dir

Qt = QtCore.Qt

# The mark drawn in the header. Beside the application first and in
# the bundle after -- the order config.asset() uses -- so a re-branded
# logo can be dropped in next to the exe without a rebuild.
LOGO_FILE = "vikbot-logo.png"

# RGB here, unlike the rest of the app: this is Qt's side of the fence.
INK = "#1a1613"
PANEL = "#2e2823"
LINE = "#4e463e"
TEXT = "#f0eeec"
MUTED = "#9aa2aa"
ACCENT = "#379ded"
OK = "#40b060"
WARN = "#f0c44a"
BAD = "#e85454"

STATES = {
    "running": ("RUNNING", OK),
    "reading": ("READING", WARN),
    "rewind": ("REWIND", BAD),
    # Not a rewind: nothing on the coil is going to fix it, the sheet has to
    # be changed. Different word, so the operator is not sent to the winder.
    "mismatch": ("WRONG SHEET", BAD),
    # Labels are going past and not one of them reads.
    "unread": ("NOT READING", BAD),
    # A label came past with its code or its logo missing.
    "incomplete": ("BAD LABEL", BAD),
    "idle": ("IDLE", MUTED),
    # No sheet has been chosen yet, so there is nothing to check the roll
    # against and START does nothing. Its own word, because IDLE would say
    # the machine is ready when it is only waiting.
    "nosheet": ("NO SHEET", MUTED),
    # The winder is on hand control, so the console cannot start it.
    "manual": ("WINDER MANUAL", WARN),
}


_UNSET = object()

_STAMP_TAIL = re.compile(r"_\d{8}-\d{6}(?:-\d{3})?(?:_\d+)?$")


def _strip_stamp(name):
    """The payload half of a crop's filename, without the stamp on the end.

    The time is shown on its own line right below, and repeating it inside
    the name costs the width that the code itself needs.
    """
    return _STAMP_TAIL.sub("", name)


def _wrap_path(path):
    """A path the panel can show whole, however deep it is.

    The operator has to be able to read the folder off the screen and go and
    find it, so it is shown in full rather than shortened. A path has no
    spaces to break at, and a label only wraps at one, so a zero-width space
    goes after each separator: it draws as nothing and gives the line
    somewhere to break -- at a folder boundary, which is where a path reads
    best anyway.
    """
    path = (path or "").rstrip(os.sep)
    return path.replace(os.sep, os.sep + "\u200b")


def to_pixmap(image):
    """A QPixmap over a BGR numpy frame, with no colour conversion.

    Format_BGR888 takes OpenCV's byte order directly, and fromImage copies,
    so the worker is free to reuse the buffer the moment this returns.
    """
    if image is None:
        return None
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    qimg = QtGui.QImage(image.data, w, h, image.strides[0],
                        QtGui.QImage.Format_BGR888)
    return QtGui.QPixmap.fromImage(qimg)


class VideoView(QtWidgets.QWidget):
    """The picture, with the label overlay and the fault banner painted on it.

    Everything it draws arrives in the snapshot already worked out, in the
    coordinates of the image it was given; all this adds is the letterbox fit
    into whatever size the window happens to be.
    """

    def __init__(self, parent=None, scale=1.0):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._pixmap = None
        self._snap = {}
        pt = lambda n: max(6.0, n * scale)
        self._caption_font = QtGui.QFont("DejaVu Sans", 0, QtGui.QFont.Bold)
        self._caption_font.setPointSizeF(pt(8))
        self._tag_font = QtGui.QFont("DejaVu Sans", 0, QtGui.QFont.Bold)
        self._tag_font.setPointSizeF(pt(10))

    def set_snapshot(self, snap):
        self._snap = snap
        self._pixmap = to_pixmap(snap.get("image"))
        self.update()

    def _fit(self):
        """Where the picture sits in the widget, and at what scale."""
        if self._pixmap is None:
            return 0, 0, 1.0
        pw, ph = self._pixmap.width(), self._pixmap.height()
        k = min(self.width() / pw, self.height() / ph)
        return (self.width() - pw * k) / 2.0, (self.height() - ph * k) / 2.0, k

    # -- painting ---------------------------------------------------------
    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor("#101010"))
        if self._pixmap is None:
            p.setPen(QtGui.QColor(MUTED))
            p.drawText(self.rect(), Qt.AlignCenter, "waiting for the camera")
            return

        ox, oy, k = self._fit()
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        p.drawPixmap(QtCore.QRectF(ox, oy,
                                   self._pixmap.width() * k,
                                   self._pixmap.height() * k), self._pixmap,
                     QtCore.QRectF(self._pixmap.rect()))

        p.save()
        p.translate(ox, oy)
        p.scale(k, k)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        # Text is drawn at the widget's scale, not the picture's, so a
        # stretched window does not give bloated captions.
        self._inv = 1.0 / k if k else 1.0
        self._draw_parts(p)
        self._draw_boxes(p)
        self._draw_tags(p)
        self._draw_fault(p)
        p.restore()

    def _scaled(self, font):
        """A font that comes out the requested size on screen whatever the
        picture has been scaled to."""
        f = QtGui.QFont(font)
        f.setPointSizeF(max(font.pointSizeF() * self._inv, 1.0))
        return f

    def _draw_parts(self, p):
        """The qr and the logo the model found inside each label.

        Thin, uncaptioned and under the label boxes: they are there to show
        what the detector has hold of, not to be read. A label outlined with
        no logo box inside it is the picture of the fault this app stops for.
        """
        p.setBrush(Qt.NoBrush)
        for x1, y1, x2, y2, rgb in self._snap.get("parts", ()):
            p.setPen(QtGui.QPen(QtGui.QColor(*rgb), 1 * self._inv))
            p.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))

    def _draw_boxes(self, p):
        """One rectangle per label, in the colour of whatever decoded it, and
        the payload on a filled strip beneath."""
        font = self._scaled(self._caption_font)
        m = QtGui.QFontMetrics(font)
        for x1, y1, x2, y2, rgb, caption in self._snap.get("boxes", ()):
            colour = QtGui.QColor(*rgb)
            p.setPen(QtGui.QPen(colour, 2 * self._inv))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QtCore.QRectF(x1, y1, x2 - x1, y2 - y1))
            if not caption:
                continue
            pad = 4 * self._inv
            w = m.horizontalAdvance(caption) + 2 * pad
            h = m.height() + pad
            top = y2 + 2 * self._inv
            if top + h > self.height() * self._inv:
                top = y1 + 2 * self._inv          # no room below: go inside
            box = QtCore.QRectF(x1, top, w, h)
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            p.drawRect(box)
            p.setPen(QtGui.QColor("#000000"))
            p.setFont(font)
            p.drawText(box.adjusted(pad, 0, 0, 0),
                       Qt.AlignVCenter | Qt.AlignLeft, caption)

    def _draw_tags(self, p):
        """The heavier outline and the caption on a label the fault is about
        -- which is the answer to 'where is it' while the coil is wound back."""
        font = self._scaled(self._tag_font)
        m = QtGui.QFontMetrics(font)
        for x1, y1, x2, y2, rgb, caption in self._snap.get("tags", ()):
            colour = QtGui.QColor(*rgb)
            p.setPen(QtGui.QPen(colour, 4 * self._inv))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QtCore.QRectF(x1 - 3 * self._inv, y1 - 3 * self._inv,
                                     x2 - x1 + 6 * self._inv,
                                     y2 - y1 + 6 * self._inv))
            if not caption:
                continue          # outlined only: the colour is the message
            pad = 5 * self._inv
            w = m.horizontalAdvance(caption) + 2 * pad
            box = QtCore.QRectF(x1, y1, w, m.height() + pad)
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            p.drawRect(box)
            p.setPen(QtGui.QColor("#000000"))
            p.setFont(font)
            p.drawText(box.adjusted(pad, 0, 0, 0),
                       Qt.AlignVCenter | Qt.AlignLeft, caption)

    def _draw_fault(self, p):
        """A red border round the picture, and nothing else.

        The headline and the detail used to be written across the video,
        over the very labels they were about. They live in the panel down the
        left now; all the picture carries is the border, which is what makes
        a stopped screen unmistakable from across the machine.
        """
        if not self._snap.get("banner") and not self._snap.get("lines"):
            return
        w, h = self._pixmap.width(), self._pixmap.height()
        p.setPen(QtGui.QPen(QtGui.QColor(BAD), 6 * self._inv))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QtCore.QRectF(2, 2, w - 4, h - 4))


class ToggleSwitch(QtWidgets.QAbstractButton):
    """A two-position selector, drawn as one.

    A pushbutton that stays down says "I have been pressed"; a switch says
    "the machine is in this position", which is the thing that matters when
    the position decides whether a relay is closed. It is painted rather
    than styled because Qt has no switch, and a checkbox with a stylesheet
    over it still reads as a checkbox at arm's length across a machine.

    The two labels sit inside the track, both always legible, with the knob
    over the one in force -- so it can be read from the far side of the
    winder without having to remember which way round on means.
    """

    def __init__(self, off_text, on_text, parent=None, scale=1.0,
                 on_colour=OK, off_colour=WARN):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self._texts = (off_text, on_text)
        self._colours = (off_colour, on_colour)
        self._k = scale
        self.setFixedHeight(int(44 * scale))
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Fixed)

    def paintEvent(self, _event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = QtCore.QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = r.height() / 2.0

        p.setPen(Qt.NoPen)
        p.setBrush(QtGui.QColor(PANEL))
        p.drawRoundedRect(r, radius, radius)

        # The knob covers its half of the track, and carries the colour.
        half = QtCore.QRectF(r.left() + (r.width() / 2.0 if self.isChecked()
                                         else 0.0),
                             r.top(), r.width() / 2.0, r.height())
        p.setBrush(QtGui.QColor(self._colours[bool(self.isChecked())]))
        p.drawRoundedRect(half.adjusted(2, 2, -2, -2), radius - 2, radius - 2)

        font = QtGui.QFont("DejaVu Sans", 0, QtGui.QFont.Bold)
        font.setPointSizeF(max(8.0 * self._k, 6.0))
        font.setLetterSpacing(QtGui.QFont.PercentageSpacing, 105)
        p.setFont(font)
        for i, text in enumerate(self._texts):
            side = QtCore.QRectF(r.left() + i * r.width() / 2.0, r.top(),
                                 r.width() / 2.0, r.height())
            live = bool(self.isChecked()) == bool(i)
            p.setPen(QtGui.QColor("#12100e" if live else MUTED))
            p.drawText(side, Qt.AlignCenter, text)

    def sizeHint(self):
        return QtCore.QSize(int(150 * self._k), int(44 * self._k))


class ColumnBox(QtWidgets.QComboBox):
    """A drop-down that looks like a drop-down.

    Qt draws a combo's arrow from the widget style, and under this dark
    stylesheet what arrives is a dark arrow on a dark field -- so the
    selector reads as a label with a column name written in it, and nobody
    presses it. Painting the chevron here rather than styling it keeps the
    app one file with no image to ship, and lets it follow the text: the
    same grey as the caption, dimmer while the line is running and the
    selector is locked.

    The chevron is drawn over the styled frame rather than instead of it, so
    the background, border and text all stay in the stylesheet with
    everything else.
    """

    def __init__(self, parent=None, scale=1.0):
        super().__init__(parent)
        self._k = scale
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(int(32 * scale))
        # The popup is a top-level window in its own right, not a child of
        # this widget, so `QComboBox QAbstractItemView` in the main
        # stylesheet never reaches it: the list came up in the stock palette
        # with the column already in force unmarked, which is the one thing
        # somebody opening it needs to see. Styling the view itself is what
        # makes the rules land.
        view = QtWidgets.QListView()
        view.setStyleSheet(f"""
            QListView {{ background: {PANEL}; color: {TEXT};
                         border: 1px solid {LINE}; outline: 0;
                         padding: {int(2 * scale)}px;
                         font-size: {int(11 * scale)}px;
                         font-weight: 600; }}
            QListView::item {{ min-height: {int(26 * scale)}px;
                               padding: 0 {int(8 * scale)}px;
                               border-radius: {int(3 * scale)}px; }}
            QListView::item:hover {{ background: #3a332c; }}
            QListView::item:selected {{ background: {ACCENT};
                                        color: #12100e; }}
        """)
        self.setView(view)
        # A combo ships with a delegate that measures rows its own way and
        # ignores what the stylesheet asks for, which draws the list with its
        # rows overlapping and the text clipped. The standard delegate takes
        # the metrics from the sheet, so the rows come out the height they
        # were asked to be.
        self.setItemDelegate(QtWidgets.QStyledItemDelegate(self))
        self._row_h = int(28 * scale)

    def addItems(self, texts):
        """Add the columns, and say how tall a row is while doing it.

        The popup sizes itself from the model, not from the stylesheet, so a
        row height asked for in QSS alone leaves the list the height Qt would
        have made it anyway -- with the rows drawn taller than the space they
        were given and every line of text overlapping the next. The size hint
        is the part the popup actually reads.
        """
        super().addItems(texts)
        for i in range(self.count()):
            self.setItemData(i, QtCore.QSize(0, self._row_h),
                             Qt.SizeHintRole)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        colour = QtGui.QColor(MUTED if self.isEnabled() else "#5a534b")
        pen = QtGui.QPen(colour, max(1.6, 1.8 * self._k))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        arm = max(3.5, 4.0 * self._k)
        cx = self.width() - max(12.0, 13.0 * self._k)
        cy = self.height() / 2.0
        path = QtGui.QPainterPath()
        path.moveTo(cx - arm, cy - arm / 2.0)
        path.lineTo(cx, cy + arm / 2.0)
        path.lineTo(cx + arm, cy - arm / 2.0)
        p.drawPath(path)
        p.end()


class CameraDialog(QtWidgets.QDialog):
    """Exposure, gain and brightness, over the live picture.

    Not modal, and deliberately: the only way to set these is to watch the
    labels while you change them. The dialog sits to one side, the video
    keeps running behind it, and every change goes to the camera as it is
    made.

    Stepped rather than dragged. A slider across a range of a thousand puts
    every value within a pixel or two of three others, which is no way to
    settle on one -- and settling on one is the whole job: exposure is
    tuned until the codes read and then left alone. The arrows move it by
    exactly one, the box takes a number typed straight in, and the wheel
    works on it for the coarse hunt.

    It builds itself from what the camera says it has. A control the device
    does not offer gets no row, and the limits are the device's own
    (narrowed by config.json) rather than anything written here.
    """

    changed = QtCore.pyqtSignal(str, int)      # control name, new value
    reset = QtCore.pyqtSignal()

    LABELS = {"exposure": "Exposure", "gain": "Gain",
              "brightness": "Brightness"}
    ORDER = ("exposure", "gain", "brightness")

    def __init__(self, ranges, values, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.setStyleSheet(
            f"QDialog {{ background: {INK}; }}"
            f"QLabel {{ color: {TEXT}; }}"
            f"QLabel#hint {{ color: {MUTED}; }}"
            f"QSpinBox {{ color: {TEXT}; background: {PANEL}; "
            f"border: 1px solid {LINE}; padding: 6px 4px; "
            f"font-family: monospace; font-size: 16px; }}"
            f"QSpinBox:focus {{ border-color: {ACCENT}; }}"
            # The arrows are the point of this control, so they are given
            # something to aim at rather than Qt's default few pixels.
            f"QSpinBox::up-button, QSpinBox::down-button {{ width: 26px; "
            f"background: {LINE}; border: none; }}"
            f"QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ "
            f"background: {ACCENT}; }}"
            f"QPushButton {{ color: {TEXT}; background: {PANEL}; "
            f"border: 1px solid {LINE}; padding: 6px 14px; }}"
            f"QPushButton:hover {{ border-color: {ACCENT}; }}")

        self._rows = {}
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(18, 16, 18, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        row = 0
        for name in self.ORDER:
            spec = ranges.get(name)
            if not spec:
                continue                       # this camera has no such knob
            title = QtWidgets.QLabel(self.LABELS.get(name, name.title()))
            title.setFont(QtGui.QFont("", 11, QtGui.QFont.DemiBold))

            box = QtWidgets.QSpinBox()
            box.setRange(int(spec["min"]), int(spec["max"]))
            box.setSingleStep(int(spec.get("step", 1)) or 1)
            box.setValue(int(values.get(name, spec.get("default", 0))))
            box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            box.setFixedWidth(140)
            # Typing 1000 means passing through 1, 10 and 100, and each of
            # those would otherwise go to the camera as it was typed. The
            # value is sent when the digits stop, not while they arrive.
            box.setKeyboardTracking(False)
            box.setAccelerated(True)           # held arrows speed up

            ends = QtWidgets.QLabel(f"{spec['min']} – {spec['max']}")
            ends.setObjectName("hint")

            grid.addWidget(title, row, 0)
            grid.addWidget(box,   row, 1)
            grid.addWidget(ends,  row, 2)
            row += 1

            def emit(value, name=name):
                self.changed.emit(name, int(value))

            box.valueChanged.connect(emit)
            self._rows[name] = (box, spec)

        if not self._rows:
            grid.addWidget(QtWidgets.QLabel(
                "This camera offers no adjustable exposure, gain or "
                "brightness."), 0, 0, 1, 3)

        note = QtWidgets.QLabel("Arrows step by one; a number can be typed "
                                "straight in. Every change reaches the camera "
                                "at once and is remembered for next time.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        grid.addWidget(note, row, 0, 1, 3)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        if self._rows:
            back = QtWidgets.QPushButton("Camera defaults")
            back.clicked.connect(self._defaults)
            buttons.addWidget(back)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        grid.addLayout(buttons, row + 1, 0, 1, 3)

    def _defaults(self):
        """Back to what the camera itself calls normal."""
        for _name, (box, spec) in self._rows.items():
            box.setValue(int(spec.get("default", box.value())))

    def update_values(self, values):
        """Show what the camera actually holds, without re-emitting it."""
        for name, (box, _spec) in self._rows.items():
            if name not in values:
                continue
            value = int(values[name])
            if value == box.value():
                continue
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)


class InspectorWindow(QtWidgets.QMainWindow):
    """The console. Emits `command(name, argument)` and nothing else."""

    command = QtCore.pyqtSignal(str, object)
    _incoming = QtCore.pyqtSignal(object)

    # Only the diagnostics toggle. Everything that moves the machine or
    # repoints the run is a button you have to look at and click: a stray
    # keypress on a console standing next to a winder must not be able to
    # stop the line, and must not be able to pop a file dialog over it
    # either. F11 is here because it is how you get out of full screen.
    # Nothing on a bare letter. This console stands next to a winder and
    # gets leaned on, brushed past and wiped down, and a single keystroke
    # that opens a window over the live picture is a keystroke that will
    # happen by accident. Both of these are chords instead, which nothing
    # short of a deliberate three-finger press produces.
    #
    #   Ctrl+Alt+E   the camera: exposure, gain, brightness
    #   Ctrl+Alt+W   the diagnostics readout
    #
    # F11 stays a bare key: it only changes the size of the window, and it
    # is the one binding everybody already expects to find.
    KEYS = {Qt.Key_F11: "fullscreen"}
    CHORDS = {"Ctrl+Alt+E": "camera", "Ctrl+Alt+W": "debug"}

    def __init__(self, title=" Label Inspect"):
        super().__init__()
        self.setWindowTitle(title)

        # Sized from the screen it is actually on, not from a number written
        # here: the same build runs on the machine's panel PC and on a
        # desk, and neither should get a window laid out for the other.
        screen = QtGui.QGuiApplication.primaryScreen()
        area = (screen.availableGeometry() if screen
                else QtCore.QRect(0, 0, 1280, 800))
        self._area = area
        # Type and padding grow with the screen so a 4K panel does not end up
        # with 11px captions, held between sane limits either way. The
        # divisor was set when the right-hand column held half of what it
        # holds now, and type sized for 900 lines of screen ran the foot of
        # that column off the bottom of a 1080 panel -- so it is scaled a
        # little more gently, and the column scrolls if even that is not
        # enough.
        self._k = max(0.8, min(area.height() / 1000.0, 1.8))
        self._left_w = int(max(280, min(area.width() * 0.20, 460)))
        self._side_w = int(max(210, min(area.width() * 0.16, 380)))
        self.resize(int(area.width() * 0.9), int(area.height() * 0.9))
        # None rather than a real value, so the first snapshot always applies
        # itself: the buttons start in whatever state the constructor left
        # them, and it is the first update that puts them right.
        self._debug = None
        self._configurable = None
        self._state = None
        self._meta = None
        self._sheet_dir = ""
        self._out_dir = ""
        self._capture_dir = ""
        self._recent = []
        self._running = False
        self._loaded = None
        self._winder_auto = None
        self._reverse = None
        # _UNSET, not None: the first snapshot has to draw the switch even
        # when what it says happens to match the default.
        self._format = _UNSET
        self._counts = None
        # Up while a snapshot is being written into the ups tick boxes, so
        # their own toggled signal does not bounce that straight back at the
        # machine as though the operator had clicked it.
        self._ups_setting = False
        # What each up was pointing at before the current click, so a
        # duplicate can be swapped back into the column this one just left.
        self._ups_map = {}
        # Not None: None is a real value here -- it is what a folder with
        # nothing in it reports -- and starting on it would skip the first
        # update and leave the card sitting on its placeholder.
        self._last_saved = _UNSET
        self._fault_text = None
        # What the camera says it can do and where it is set, kept fresh off
        # the snapshot so the dialog opens showing the truth rather than
        # whatever it was told the last time it was open.
        self._camera_ranges = {}
        self._camera_values = {}
        self._camera_dialog = None
        # Set False to let the window close on a single click, the way an
        # ordinary application does.
        self.locked = True
        self._build(title)
        self._bind_chords()
        # Queued by default across threads, so the worker can post a snapshot
        # without touching a widget.
        self._incoming.connect(self._apply, Qt.QueuedConnection)

    # -- construction -----------------------------------------------------
    def _build(self, title):
        k = self._k          # every size below is in screen-scaled pixels
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {INK}; color: {TEXT};
                                    font-family: 'DejaVu Sans'; }}
            QLabel#title {{ font-size: {int(21 * k)}px; font-weight: 600;
                            letter-spacing: 1px; }}
            QLabel#meta  {{ color: {MUTED}; font-size: {int(11 * k)}px; }}
            QLabel#pill  {{ font-size: {int(13 * k)}px; font-weight: 700;
                            padding: {int(6 * k)}px {int(18 * k)}px;
                            border-radius: {int(13 * k)}px; }}
            QLabel#status{{ color: {MUTED}; font-size: {int(12 * k)}px; }}
            QLabel#caption {{ color: #7d858d; font-size: {int(10 * k)}px;
                              font-weight: 700; letter-spacing: 1px; }}
            QLabel#value {{ color: {TEXT}; font-size: {int(12 * k)}px; }}
            QLabel#count {{ color: {TEXT}; font-size: {int(15 * k)}px;
                            font-weight: 600; }}
            QLabel#faulthead {{ background: {BAD}; color: #2a0505;
                                font-size: {int(15 * k)}px; font-weight: 700;
                                padding: {int(12 * k)}px;
                                border-radius: {int(6 * k)}px; }}
            QLabel#faultbody {{ font-size: {int(12 * k)}px;
                                padding: {int(12 * k)}px 2px;
                                font-family: 'DejaVu Sans Mono', monospace; }}
            QLabel#hint  {{ color: #6d757d; font-size: {int(11 * k)}px; }}
            QFrame#rule  {{ background: {LINE}; max-height: 1px;
                            border: none; }}
            QFrame#header, QFrame#footer {{ background: {INK}; }}
            QPushButton {{ font-size: {int(15 * k)}px; font-weight: 600;
                           border-radius: {int(6 * k)}px;
                           padding: {int(14 * k)}px {int(8 * k)}px;
                           border: 1px solid {LINE}; background: {PANEL};
                           color: {TEXT}; }}
            QPushButton:disabled {{ color: #6d757d; background: #262019;
                                    border-color: #3a332c; }}
            QPushButton#start {{ background: {OK}; border: none;
                                 color: #08240f; }}
            QPushButton#stop  {{ background: {BAD}; border: none;
                                 color: #2a0505; }}
            QPushButton#start:disabled, QPushButton#stop:disabled {{
                background: #3a423c; color: #79817a; }}
            QPushButton#secondary {{ font-size: {int(12 * k)}px; font-weight: 600;
                                     color: {ACCENT};
                                     padding: {int(10 * k)}px {int(8 * k)}px; }}
            /* An id selector outranks :disabled, so without this the three
               greyed-out buttons keep their blue lettering and still read as
               something you can press. */
            QPushButton#secondary:disabled {{ color: #6d757d; }}
            QScrollArea#sidescroll {{ background: transparent;
                                      border: none; }}
            QScrollBar:vertical {{ background: transparent;
                                   width: {int(9 * k)}px; margin: 0; }}
            QScrollBar::handle:vertical {{ background: {LINE};
                                           border-radius: {int(4 * k)}px;
                                           min-height: {int(40 * k)}px; }}
            QScrollBar::handle:vertical:hover {{ background: {MUTED}; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent; }}
            QPushButton#quit {{ font-size: {int(11 * k)}px;
                                font-weight: 700; color: {MUTED};
                                padding: {int(5 * k)}px {int(14 * k)}px; }}
            QPushButton#quit:hover {{ color: {BAD}; border-color: {BAD}; }}
            QCheckBox {{ color: {TEXT}; font-size: {int(12 * k)}px;
                         font-weight: 600; spacing: {int(8 * k)}px;
                         padding: {int(4 * k)}px 0; }}
            QCheckBox:disabled {{ color: #6d757d; }}
            QCheckBox::indicator {{ width: {int(18 * k)}px;
                                    height: {int(18 * k)}px;
                                    border-radius: {int(4 * k)}px;
                                    border: 1px solid {LINE};
                                    background: {PANEL}; }}
            QCheckBox::indicator:checked {{ background: {OK};
                                            border-color: {OK}; }}
            QCheckBox::indicator:disabled {{ background: #262019;
                                             border-color: #3a332c; }}
            /* A locked tick is still a tick. Qt takes the LAST matching rule
               of equal specificity, so without this one the :disabled fill
               above paints over :checked and every up reads as switched off
               the moment the machine starts -- which is exactly when the
               operator looks at them. Dimmed, not grey: it says "on, and not
               yours to change while the line runs". */
            QCheckBox::indicator:checked:disabled {{ background: #2d6b41;
                                                     border-color: #2d6b41; }}
            /* Right padding leaves the gutter the chevron is painted in;
               the lighter fill and the border are what separate a control
               that can be pressed from the captions around it. */
            QComboBox {{ background: {PANEL}; color: {TEXT};
                         border: 1px solid {LINE};
                         border-radius: {int(4 * k)}px;
                         padding: {int(4 * k)}px {int(24 * k)}px
                                  {int(4 * k)}px {int(8 * k)}px;
                         font-size: {int(11 * k)}px; font-weight: 600; }}
            QComboBox:hover {{ border-color: {ACCENT}; }}
            QComboBox:on {{ border-color: {ACCENT}; background: #3a332c; }}
            QComboBox:disabled {{ color: #6d757d; background: #201b17;
                                  border-color: #3a332c; }}
            /* The native button is turned off entirely: it paints a raised
               square on some styles and nothing at all on others, and the
               chevron is drawn by the widget either way. */
            QComboBox::drop-down {{ border: 0; width: 0; }}
            QComboBox::down-arrow {{ image: none; width: 0; height: 0; }}
            /* The open list is styled on the view itself, in ColumnBox --
               a popup is a window of its own and a descendant selector from
               here does not reach it. */
            QGroupBox {{ border: 1px solid {LINE}; border-radius: {int(6 * k)}px;
                         margin-top: {int(10 * k)}px;
                         font-size: {int(11 * k)}px; color: {MUTED}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 10px;
                                padding: 0 4px; }}
        """)

        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header(title))
        outer.addWidget(self._rule())

        middle = QtWidgets.QHBoxLayout()
        middle.setContentsMargins(0, 0, 0, 0)
        middle.setSpacing(0)
        middle.addWidget(self._fault_panel())
        self.video = VideoView(scale=self._k)
        middle.addWidget(self.video, 1)
        middle.addWidget(self._sidebar())
        outer.addLayout(middle, 1)

        outer.addWidget(self._rule())
        outer.addWidget(self._footer())
        self.setCentralWidget(root)

        # The rolling-window comparison, in a dock rather than a second
        # top-level window, so it can be put away without losing it.
        self.view_label = QtWidgets.QLabel(alignment=Qt.AlignCenter)
        self.view_label.setStyleSheet(f"background: {INK};")
        self.dock = QtWidgets.QDockWidget("Rolling window", self)
        self.dock.setWidget(self.view_label)
        self.dock.setAllowedAreas(Qt.RightDockWidgetArea |
                                  Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.hide()

    def _rule(self):
        f = QtWidgets.QFrame()
        f.setObjectName("rule")
        f.setFixedHeight(1)
        return f

    def _logo(self, height):
        """The Vikbot mark, scaled to sit with the header type.

        Artwork that has gone missing must never be the thing that stops a
        line, so a logo that cannot be found is simply not drawn and the
        header falls back to the accent mark and the title on their own.
        """
        for base in (app_dir(), bundle_dir()):
            path = os.path.join(base, LOGO_FILE)
            if os.path.exists(path):
                pix = QtGui.QPixmap(path)
                if not pix.isNull():
                    return pix.scaledToHeight(height, Qt.SmoothTransformation)
        return None

    def _header(self, title):
        bar = QtWidgets.QFrame()
        bar.setObjectName("header")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(12)

        mark = QtWidgets.QFrame()
        mark.setFixedWidth(4)
        mark.setStyleSheet(f"background: {ACCENT}; border-radius: 2px;")
        row.addWidget(mark)

        logo = self._logo(int(30 * self._k))
        if logo is not None:
            badge = QtWidgets.QLabel()
            badge.setPixmap(logo)
            row.addWidget(badge)

        self.title_label = QtWidgets.QLabel(title, objectName="title")
        row.addWidget(self.title_label)
        row.addStretch(1)

        # "STATUS" beside it, because a lone coloured lozenge reads as a
        # button you are meant to press rather than as a state you are being
        # told about.
        row.addWidget(QtWidgets.QLabel("MACHINE STATUS", objectName="caption"))
        self.pill = QtWidgets.QLabel("IDLE", objectName="pill")
        self._set_pill("idle")
        row.addWidget(self.pill)
        return bar

    def _fault_panel(self):
        """What is wrong, down the left, instead of written over the picture.

        Burning it into the video meant the operator had to read text laid
        over the very labels the text was about. Out here it has room, it is
        set at a readable size whatever the camera resolution, and the
        picture stays a picture. The panel only exists while a fault does.
        """
        self.left = QtWidgets.QWidget()
        self.left.setFixedWidth(self._left_w)
        col = QtWidgets.QVBoxLayout(self.left)
        col.setContentsMargins(14, 14, 14, 14)
        col.setSpacing(0)

        # The fault text is the only thing here that comes and goes, so it is
        # a block of its own: the column itself stays, because the controls
        # under it are the ones somebody reaches for with a hand already on
        # the machine, and a control that moves when a fault appears is a
        # control that gets pressed by mistake.
        self.fault_block = QtWidgets.QWidget()
        fault_col = QtWidgets.QVBoxLayout(self.fault_block)
        fault_col.setContentsMargins(0, 0, 0, 0)
        fault_col.setSpacing(0)
        self.fault_head = QtWidgets.QLabel("", objectName="faulthead")
        self.fault_head.setWordWrap(True)
        self.fault_body = QtWidgets.QLabel("", objectName="faultbody")
        self.fault_body.setWordWrap(True)
        self.fault_body.setTextFormat(Qt.RichText)
        self.fault_body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        fault_col.addWidget(self.fault_head)
        fault_col.addWidget(self.fault_body)
        col.addWidget(self.fault_block)
        self.fault_block.hide()

        col.addStretch(1)

        # The whole picture, as the camera saw it, whenever somebody wants
        # one: the frame worth keeping is the one on the screen when
        # something looks wrong, so unlike the folder buttons opposite this
        # one is live while the line runs.
        self.capture_btn = QtWidgets.QPushButton("CAPTURE FRAME",
                                                 objectName="secondary")
        self.capture_dir_btn = QtWidgets.QPushButton("CAPTURE FOLDER",
                                                     objectName="secondary")
        self.capture_btn.clicked.connect(
            lambda: self.command.emit("capture", None))
        self.capture_dir_btn.clicked.connect(self._choose_capture_dir)
        col.addWidget(self.capture_btn)
        col.addSpacing(6)
        col.addWidget(self.capture_dir_btn)

        # The winder at the very bottom of the column, on its own under a
        # rule. It is the one switch on this screen that moves the machine
        # rather than the software, and it is thrown all shift -- so it sits
        # where a hand falls, apart from anything it could be confused with.
        col.addSpacing(12)
        col.addWidget(self._rule())
        col.addSpacing(8)
        col.addWidget(QtWidgets.QLabel("WINDER", objectName="caption"))
        self.winder_btn = ToggleSwitch("MANUAL", "AUTO", scale=self._k,
                                       on_colour=OK, off_colour=WARN)
        self.winder_btn.clicked.connect(
            lambda on: self.command.emit("winder", bool(on)))
        col.addWidget(self.winder_btn)
        return self.left

    def _sidebar(self):
        """The right-hand column: the machine's buttons pinned, the rest
        scrolled.

        There is more in this column than a short screen has room for, and a
        panel PC is short -- so everything below the buttons sits in a
        scroll area rather than running off the bottom of the glass. START
        and STOP never scroll: whatever else is on the screen, those have to
        be under the operator's hand. CAPTURE FRAME, CAPTURE FOLDER and the
        winder switch are in the left column, which never scrolls at all.
        """
        side = QtWidgets.QWidget()
        side.setFixedWidth(self._side_w)
        outer = QtWidgets.QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        pinned = QtWidgets.QWidget()
        acts = QtWidgets.QVBoxLayout(pinned)
        acts.setContentsMargins(14, 14, 14, 0)
        acts.setSpacing(8)

        body = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(body)
        col.setContentsMargins(14, 8, 14, 14)
        col.setSpacing(8)

        self.start_btn = QtWidgets.QPushButton("START", objectName="start")
        self.stop_btn = QtWidgets.QPushButton("STOP", objectName="stop")
        self.sheet_btn = QtWidgets.QPushButton("LOAD SHEET",
                                               objectName="secondary")
        # The same sheet is loaded morning after morning, and hunting it down
        # in a file dialog every time is a job the console can do instead.
        self.recent_btn = QtWidgets.QPushButton("OPEN RECENT SHEET",
                                                objectName="secondary")
        self.out_btn = QtWidgets.QPushButton("LABEL FOLDER",
                                             objectName="secondary")
        self.start_btn.clicked.connect(lambda: self.command.emit("start", None))
        self.stop_btn.clicked.connect(lambda: self.command.emit("stop", None))
        self.sheet_btn.clicked.connect(self._choose_sheet)
        self.recent_btn.clicked.connect(self._open_recent)
        self.out_btn.clicked.connect(self._choose_output)
        for b in (self.start_btn, self.stop_btn):
            b.setFixedHeight(int(58 * self._k))
        acts.addWidget(self.start_btn)
        acts.addWidget(self.stop_btn)
        acts.addSpacing(6)
        acts.addWidget(self._rule())
        outer.addWidget(pinned)

        col.addWidget(self.sheet_btn)
        col.addWidget(self.recent_btn)
        col.addWidget(self.out_btn)

        self.run_box = QtWidgets.QGroupBox("RUN")
        run_col = QtWidgets.QVBoxLayout(self.run_box)
        run_col.setContentsMargins(10, 6, 10, 10)
        run_col.setSpacing(2)
        run_col.addWidget(QtWidgets.QLabel("SHEET", objectName="caption"))
        self.sheet_label = QtWidgets.QLabel("", objectName="value")
        self.sheet_label.setWordWrap(True)
        run_col.addWidget(self.sheet_label)
        run_col.addSpacing(8)
        run_col.addWidget(QtWidgets.QLabel("LABEL FOLDER",
                                           objectName="caption"))
        self.out_label = QtWidgets.QLabel("", objectName="value")
        self.out_label.setWordWrap(True)
        run_col.addWidget(self.out_label)
        run_col.addSpacing(8)
        run_col.addWidget(QtWidgets.QLabel("CAPTURE FOLDER",
                                           objectName="caption"))
        self.capture_label = QtWidgets.QLabel("", objectName="value")
        self.capture_label.setWordWrap(True)
        run_col.addWidget(self.capture_label)
        col.addSpacing(12)
        col.addWidget(self.run_box)

        self.debug_box = QtWidgets.QGroupBox("DIAGNOSTICS")
        inner = QtWidgets.QVBoxLayout(self.debug_box)
        inner.setContentsMargins(10, 6, 10, 10)
        self.debug_label = QtWidgets.QLabel("", objectName="meta")
        self.debug_label.setWordWrap(True)
        inner.addWidget(self.debug_label)
        col.addSpacing(10)
        col.addWidget(self.debug_box)
        self.debug_box.hide()

        # The two selectors live at the foot of the column, under everything
        # that acts, because neither of them acts: they say what the machine
        # is set to. Direction above winder because it is set once when the
        # reel goes on, while the winder is thrown all shift.
        col.addStretch(1)

        # The tally, immediately above the two selectors. It is the one part
        # of the column that answers "is this run getting anywhere?", and the
        # operator reads it at a glance from across the machine -- so the
        # numbers are set larger than the captions beside them.
        # "CODES", not "QR": a QR DATA column holds either symbol, and which
        # one a row is printed as is the row's business. The totals count
        # both, and the split under them says how the sheet divides -- shown
        # only when there is a datamatrix in it, because on an all-QR sheet
        # the line would say nothing the total has not said already.
        self.count_box = QtWidgets.QGroupBox("CODES")
        counts = QtWidgets.QGridLayout(self.count_box)
        counts.setContentsMargins(10, 6, 10, 8)
        counts.setVerticalSpacing(3)
        counts.setColumnStretch(0, 1)
        self.count_labels = {}
        for row, (key, text) in enumerate((("sheet", "IN EXCEL"),
                                           ("read", "SCANNED"),
                                           ("saved", "IMAGES SAVED"))):
            counts.addWidget(QtWidgets.QLabel(text, objectName="caption"),
                             row, 0)
            value = QtWidgets.QLabel("0", objectName="count")
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            counts.addWidget(value, row, 1)
            self.count_labels[key] = value
        self.mix_label = QtWidgets.QLabel("", objectName="meta")
        counts.addWidget(self.mix_label, 3, 0, 1, 2)
        self.mix_label.hide()
        col.addWidget(self.count_box)
        col.addSpacing(8)

        # The last crop written, directly above the two selectors. It is the
        # console's answer to "is it still reading?" -- a name that keeps
        # changing says yes far more plainly than a count going up by one.
        self.last_box = QtWidgets.QGroupBox("LAST SCANNED")
        last = QtWidgets.QVBoxLayout(self.last_box)
        last.setContentsMargins(10, 6, 10, 8)
        last.setSpacing(2)
        self.last_name = QtWidgets.QLabel("—", objectName="value")
        self.last_name.setWordWrap(True)
        self.last_time = QtWidgets.QLabel("", objectName="meta")
        # Wrapped: the line now carries the row, the up and the log file, and
        # a panel PC's sidebar is narrower than all three end to end.
        self.last_time.setWordWrap(True)
        last.addWidget(self.last_name)
        last.addWidget(self.last_time)
        col.addWidget(self.last_box)
        col.addSpacing(10)

        # Which ups across the web are being checked, one box per code
        # column the sheet has. A reel run narrower than the sheet was
        # written for -- three ups off a four-up sheet -- would otherwise
        # stop the line at every row for the label that is not on the web,
        # so the ups that are not running are ticked off here instead. It
        # starts empty and hidden: how many ups there are is the sheet's
        # to say, and no sheet is loaded yet.
        self.ups_box = QtWidgets.QGroupBox("UPS CHECKED")
        self.ups_grid = QtWidgets.QGridLayout(self.ups_box)
        self.ups_grid.setContentsMargins(10, 6, 10, 8)
        self.ups_grid.setVerticalSpacing(2)
        self.ups_boxes = []
        self.ups_combos = []
        # The mapping written out in words under the selectors. Two ways of
        # saying the same thing on purpose: the selectors are how it is set,
        # this is how it is read back at a glance from across the machine --
        # and it is the line somebody photographs when they want to record
        # how the job was set up.
        self.ups_summary = QtWidgets.QLabel("", objectName="meta")
        self.ups_summary.setWordWrap(True)
        col.addWidget(self.ups_box)
        col.addWidget(self.ups_summary)
        self.ups_box.hide()
        self.ups_summary.hide()
        col.addSpacing(10)

        # What the camera sends. Above the check direction because it is set
        # once for a job, like the direction is, and unlike the winder it does
        # not act on the machine -- it changes what the pictures are made of.
        self.format_caption = QtWidgets.QLabel("CAMERA FORMAT",
                                               objectName="caption")
        col.addWidget(self.format_caption)
        self.format_switch = ToggleSwitch("MJPG", "YUYV", scale=self._k,
                                          on_colour=ACCENT, off_colour=ACCENT)
        self.format_switch.clicked.connect(
            lambda on: self.command.emit("format", bool(on)))
        col.addWidget(self.format_switch)
        # The rates each one is worth at this size, from the camera itself --
        # a number written here would be a number about somebody else's
        # camera. This is what makes the choice a choice and not a guess.
        self.format_rates = QtWidgets.QLabel("", objectName="meta")
        self.format_rates.setWordWrap(True)
        col.addWidget(self.format_rates)
        col.addSpacing(10)

        col.addWidget(QtWidgets.QLabel("CHECK DIRECTION IN EXCEL",
                                       objectName="caption"))
        self.dir_switch = ToggleSwitch("FORWARD", "REVERSE", scale=self._k,
                                       on_colour=ACCENT, off_colour=ACCENT)
        self.dir_switch.clicked.connect(
            lambda on: self.command.emit("direction", bool(on)))
        col.addWidget(self.dir_switch)

        self.side_scroll = scroll = QtWidgets.QScrollArea(
            objectName="sidescroll")
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)      # the column keeps the full width
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # Dragged as well as wheeled: this is a touchscreen, and a scrollbar
        # ten pixels wide is not something a finger finds. QScroller only
        # takes over past a drag threshold, so a press that does not move is
        # still a press on the button under it.
        QtWidgets.QScroller.grabGesture(
            scroll.viewport(), QtWidgets.QScroller.LeftMouseButtonGesture)
        outer.addWidget(scroll, 1)
        return side

    def _footer(self):
        bar = QtWidgets.QFrame()
        bar.setObjectName("footer")
        row = QtWidgets.QHBoxLayout(bar)
        row.setContentsMargins(16, 8, 16, 8)
        self.status = QtWidgets.QLabel("Press START to read the labels and "
                                       "run.", objectName="status")
        row.addWidget(self.status)
        row.addStretch(1)
        row.addWidget(QtWidgets.QLabel("[F11] FULL SCREEN", objectName="hint"))
        # In the corner the window's own close button would be in, because
        # in full screen there is no title bar to put one in -- and a panel
        # PC with no keyboard has no other way out of the application.
        self.quit_btn = QtWidgets.QPushButton("QUIT", objectName="quit")
        self.quit_btn.clicked.connect(self._quit_clicked)
        row.addSpacing(14)
        row.addWidget(self.quit_btn)
        return bar

    # -- input ------------------------------------------------------------
    def _choose_sheet(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose the validation sheet", self._sheet_dir,
            "Excel workbook (*.xlsx *.xlsm);;All files (*)")
        if path:
            self.command.emit("sheet", path)

    def _choose_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the folder for the label crops", self._out_dir)
        if path:
            self.command.emit("labeldir", path)

    def _choose_capture_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the folder for captured frames", self._capture_dir)
        if path:
            self.command.emit("capturedir", path)

    def _open_recent(self):
        """The sheets loaded before, as a menu under the button.

        Built fresh on each click rather than kept in step with the snapshot:
        it is opened by hand a few times a shift, and a menu that is only
        ever right at the moment it is shown cannot go stale.
        """
        menu = self._recent_menu()
        menu.exec_(self.recent_btn.mapToGlobal(
            QtCore.QPoint(0, self.recent_btn.height())))

    def _recent_menu(self):
        menu = QtWidgets.QMenu(self)
        for path in self._recent:
            act = menu.addAction(os.path.basename(path))
            act.setToolTip(path)
            act.triggered.connect(
                lambda _checked=False, p=path: self.command.emit("sheet", p))
        if menu.isEmpty():
            menu.addAction("No other sheets loaded yet").setEnabled(False)
        return menu

    def _build_ups(self, count):
        """One row per up the sheet has: the tick box, and the sheet column
        that up is checked against.

        Rebuilt rather than hidden and shown, because how many there are is
        a property of the sheet that is loaded and a sheet can be swapped
        for a wider or narrower one without the console restarting.

        A row rather than two boxes to a line, which is what this was before
        the selector arrived: an up and the column it answers to have to be
        readable as one thing, and side by side they are.
        """
        for widget in self.ups_boxes + self.ups_combos:
            self.ups_grid.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.ups_boxes, self.ups_combos = [], []
        for i in range(count):
            box = QtWidgets.QCheckBox(f"UP{i + 1}")
            box.setChecked(True)
            box.setEnabled(self._configurable is not False)
            box.toggled.connect(self._ups_changed)
            self.ups_grid.addWidget(box, i, 0)
            self.ups_boxes.append(box)

            combo = ColumnBox(scale=self._k)
            combo.addItems([f"QR DATA{c + 1}" for c in range(count)])
            combo.setCurrentIndex(i)
            combo.setEnabled(self._configurable is not False)
            combo.currentIndexChanged.connect(
                lambda _idx, up=i: self._map_changed(up))
            self.ups_grid.addWidget(combo, i, 1)
            self.ups_combos.append(combo)
        self.ups_grid.setColumnStretch(1, 1)
        self.ups_box.setVisible(bool(count))
        self.ups_summary.setVisible(bool(count))
        self._say_map()

    def _map_changed(self, up):
        """One up was pointed at a different column.

        The column it has just taken is given up by whoever held it, so the
        mapping is always a permutation and never has two ups reading the
        same cell -- there is one label at each position on the web, and a
        mapping that says otherwise is one nobody meant to make. Swapping
        also means the operator can never be left half way through setting
        one up, with a state the machine would refuse.
        """
        if self._ups_setting:
            return                  # a snapshot being written in, not a click
        want = self.ups_combos[up].currentIndex()
        self._ups_setting = True
        for other, combo in enumerate(self.ups_combos):
            if other != up and combo.currentIndex() == want:
                combo.setCurrentIndex(self._ups_map.get(up, up))
                break
        self._ups_map = {i: c.currentIndex()
                         for i, c in enumerate(self.ups_combos)}
        self._ups_setting = False
        self._say_map()
        self.command.emit("upsmap", [c.currentIndex()
                                     for c in self.ups_combos])

    def _say_map(self):
        """The mapping as one line of text, with the ups that are off named."""
        parts = []
        for i, combo in enumerate(self.ups_combos):
            on = self.ups_boxes[i].isChecked() if i < len(self.ups_boxes) else True
            parts.append(f"UP{i + 1}\u2192C{combo.currentIndex() + 1}"
                         + ("" if on else " off"))
        self.ups_summary.setText("  ".join(parts))

    def _ups_changed(self):
        if self._ups_setting:
            return                  # a snapshot being written in, not a click
        self._say_map()
        wanted = [i for i, b in enumerate(self.ups_boxes) if b.isChecked()]
        if not wanted:
            # Checking no ups at all checks nothing, and a run that checks
            # nothing passes every row it is given -- so the last tick will
            # not come off. The operator turns off the ups the reel is not
            # running, not all of them.
            box = self.sender()
            if isinstance(box, QtWidgets.QCheckBox):
                self._ups_setting = True
                box.setChecked(True)
                self._ups_setting = False
            return
        self.command.emit("ups", wanted)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showMaximized()     # never trap anyone in full screen
            return
        name = self.KEYS.get(event.key())
        if name is None:
            return super().keyPressEvent(event)
        if name == "fullscreen":
            self.showMaximized() if self.isFullScreen() \
                else self.showFullScreen()
        else:
            self._fire(name)

    def _fire(self, name):
        """What a shortcut does, whichever way it was pressed."""
        if name == "camera":
            self.open_camera_dialog()
        else:
            self.command.emit(name, None)

    def _bind_chords(self):
        """The two chords, as application shortcuts.

        QShortcut rather than keyPressEvent, so they still fire while the
        camera window has the focus -- the operator opens it, adjusts, and
        wants the same chord to shut it again without hunting for the
        console first.
        """
        for sequence, name in self.CHORDS.items():
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(lambda name=name: self._fire(name))

    # -- the camera sliders -----------------------------------------------
    def open_camera_dialog(self):
        """Open the camera controls, or shut them again."""
        if self._camera_dialog is not None:
            self._camera_dialog.close()      # the same chord puts it away
            return
        if not self._camera_ranges:
            # Said in the status bar, not in a message box. A modal dialog
            # stops the GUI thread until somebody clicks it, and this
            # console can be several metres from the nearest hand -- the
            # picture would sit frozen over a moving web in the meantime.
            self.status.setText("This camera's exposure, gain and brightness "
                                "cannot be reached from here.")
            return
        dialog = CameraDialog(self._camera_ranges, self._camera_values, self)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.changed.connect(
            lambda name, value: self.command.emit("camera", (name, value)))
        dialog.destroyed.connect(self._camera_dialog_gone)
        self._camera_dialog = dialog
        dialog.show()

    def _camera_dialog_gone(self, *_):
        self._camera_dialog = None

    # -- closing ----------------------------------------------------------
    def _may_close(self):
        """Whether the operator really meant to shut the inspection down.

        The window sits on a machine that is winding a coil, and the close
        button is a few pixels from nothing in particular. Closing it stops
        the line and shuts the run's books, so a stray click must not do it:
        while the machine is running or held on a fault the answer is simply
        no, and even idle it has to be confirmed.
        """
        if self._running:
            QtWidgets.QMessageBox.warning(
                self, "Label Inspector",
                "The machine is running.\n\n"
                "Press STOP first, then close.",
                QtWidgets.QMessageBox.Ok)
            return False
        if self._state == "rewind":
            QtWidgets.QMessageBox.warning(
                self, "Label Inspector",
                "A row is still held for re-inspection.\n\n"
                "Wind the coil back until it reads, or press START to record "
                "it as a defect, before closing.",
                QtWidgets.QMessageBox.Ok)
            return False
        answer = QtWidgets.QMessageBox.question(
            self, "Label Inspector",
            "Close Label Inspector?\n\n"
            "The workbook is written out on the way, so nothing already "
            "checked is lost.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        return answer == QtWidgets.QMessageBox.Yes

    def _quit_clicked(self):
        """Shut the application down from the screen, whatever the line is
        doing.

        The window's own close button refuses while the machine is running
        or a row is held: a stray click on the chrome must not be able to
        stop a coil mid-inspection. This is the way out that always works,
        so it says plainly what it is about to do and then does it. Nothing
        is lost by it -- the shutdown stops the machine, drops the winder
        relay and writes the workbook out, exactly as an idle close does.
        """
        if self._running or self._state in ("rewind", "mismatch", "unread",
                                            "incomplete"):
            question = ("Quit while the machine is running?\n\n"
                        "The line will be stopped and the winder relay "
                        "released. What has been checked so far is written "
                        "out."
                        if self._running else
                        "Quit with a row still held?\n\n"
                        "The row will be left unchecked. What has been "
                        "checked so far is written out.")
        else:
            question = ("Quit Label Inspector?\n\n"
                        "The workbook is written out on the way, so nothing "
                        "already checked is lost.")
        answer = QtWidgets.QMessageBox.question(
            self, "Label Inspector", question,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        # Asked and answered: the close itself must not ask again, and must
        # not refuse for the state this button exists to get out of.
        was, self.locked = self.locked, False
        if not self.close():
            self.locked = was            # refused after all; leave it as found

    def closeEvent(self, event):
        if self.locked and not self._may_close():
            event.ignore()
            return
        self.command.emit("quit", None)
        event.accept()

    # -- output -----------------------------------------------------------
    def post(self, snap):
        """Called from the capture thread. Hands the snapshot to the GUI
        thread through a queued signal and returns at once."""
        self._incoming.emit(snap)

    def _set_pill(self, state):
        label, colour = STATES.get(state, STATES["idle"])
        self.pill.setText(label)
        if state in ("idle", "nosheet"):
            self.pill.setStyleSheet(f"color: {MUTED}; background: {PANEL}; "
                                    f"border: 1px solid {LINE};")
        else:
            self.pill.setStyleSheet(f"color: {INK}; background: {colour};")

    def _show_fault(self, snap):
        """The fault panel: the headline in a red block, the detail under it.

        Built as one rich-text block rather than a stack of labels because
        the lines change shape between the two kinds of fault, and rebuilding
        a layout sixty times a second would be silly.
        """
        banner = snap.get("banner") or ""
        lines = snap.get("lines") or ()
        if not banner and not lines:
            if self.fault_block.isVisible():
                self.fault_block.hide()
            self._fault_text = None
            return

        body = "<br>".join(
            f'<span style="color:#{r:02x}{g:02x}{b:02x}">'
            f'{html.escape(text)}</span>' for text, (r, g, b) in lines)
        if (banner, body) == self._fault_text:
            return
        self._fault_text = (banner, body)
        self.fault_head.setText(banner)
        self.fault_body.setText(body)
        if not self.fault_block.isVisible():
            self.fault_block.show()

    def _apply(self, snap):
        self.video.set_snapshot(snap)

        state = snap.get("state", "idle")
        self._running = state in ("running", "reading")
        # Two things gate START besides the machine's own state, and both
        # are tracked separately from it: a sheet to check the roll against,
        # and the winder in AUTO. Either can change while the state stays
        # exactly where it was -- idle -- and the button would never be
        # re-enabled if it were only watching the state.
        loaded = bool(snap.get("loaded", True))
        auto = bool(snap.get("winder_auto", True))
        if state != self._state or loaded != self._loaded \
                or auto != self._winder_auto:
            self._state, self._loaded, self._winder_auto = state, loaded, auto
            self._set_pill(state if loaded else "nosheet")
            self.start_btn.setEnabled(
                loaded and auto and state in ("idle", "rewind", "mismatch",
                                              "unread", "incomplete"))
            self.stop_btn.setEnabled(self._running)
            self.winder_btn.setChecked(auto)

        camera = snap.get("camera") or {}
        self._camera_ranges = camera.get("ranges") or {}
        self._camera_values = camera.get("values") or {}

        # Redrawn only when a number actually moves. This runs on every
        # frame, and setText on an unchanged label still costs a repaint.
        counts = snap.get("counts") or {}
        if counts != self._counts:
            self._counts = dict(counts)
            for key, label in self.count_labels.items():
                label.setText(f"{counts.get(key, 0):,}")
            dm = counts.get("dm", 0)
            self.mix_label.setVisible(bool(dm))
            if dm:
                self.mix_label.setText(f"{counts.get('qr', 0):,} QR  ·  "
                                       f"{dm:,} DATAMATRIX")

        last = snap.get("last_saved")
        if last != self._last_saved:
            self._last_saved = last
            if not last:
                self.last_name.setText("—")
                self.last_time.setText("nothing saved yet")
            else:
                # The id the code was filed under -- the part the operator
                # reads and the part the crop is named after.
                name = os.path.splitext(last.get("name") or "")[0]
                self.last_name.setText(_strip_stamp(name) or name)
                # Where it belongs and which log file it went into, under
                # the id. A code on its own says what was read; the row and
                # the up say where it was, which is what somebody standing
                # at the machine is actually asking.
                where = " · ".join(
                    str(bit) for bit in
                    (last.get("when"),
                     f"row {last['row']}" if last.get("row") not in (None, "")
                     else None,
                     last.get("up"),
                     "DATAMATRIX" if last.get("kind") == "datamatrix" else None,
                     last.get("file"))
                    if bit)
                if where:
                    self.last_time.setText(where)
                else:
                    # No log running (--no-scan-log): fall back to the crop
                    # saver, which knows only when it wrote the file.
                    when = last.get("at")
                    self.last_time.setText(
                        time.strftime("%H:%M:%S", time.localtime(when))
                        if when else "")

        # The ups the machine is checking. Compared against the boxes
        # themselves rather than against the last snapshot: a tick the
        # machine refused -- the line started between the click and the
        # command being picked up -- has to go back on, and the snapshot
        # that says so is identical to the one before it.
        ups = snap.get("ups") or {}
        count = int(ups.get("count") or 0)
        on = tuple(sorted(ups.get("checked") or ()))
        if count != len(self.ups_boxes):
            self._build_ups(count)
        if on != tuple(i for i, b in enumerate(self.ups_boxes)
                       if b.isChecked()):
            self._ups_setting = True
            for i, box in enumerate(self.ups_boxes):
                box.setChecked(i in on)
            self._ups_setting = False
            self._say_map()

        colmap = list(ups.get("map") or range(len(self.ups_combos)))
        if colmap != [c.currentIndex() for c in self.ups_combos]:
            self._ups_setting = True
            for i, combo in enumerate(self.ups_combos):
                if i < len(colmap):
                    combo.setCurrentIndex(colmap[i])
            self._ups_map = {i: c.currentIndex()
                             for i, c in enumerate(self.ups_combos)}
            self._ups_setting = False
            self._say_map()

        fmt = snap.get("format") or {}
        if fmt != self._format:
            self._format = fmt
            live = bool(fmt.get("live", True))
            for widget in (self.format_caption, self.format_switch,
                           self.format_rates):
                widget.setVisible(live)
            if live:
                self._ups_setting = True     # a snapshot, not a click
                self.format_switch.setChecked(fmt.get("name") == "YUYV")
                self._ups_setting = False
                rates = fmt.get("rates") or {}
                self.format_rates.setText(
                    f"{fmt.get('name', '')} at {fmt.get('fps', '')}/s"
                    + ("   ·   " + ", ".join(
                        f"{k} {v:g}/s" for k, v in sorted(rates.items()))
                       if rates else ""))

        reverse = bool(snap.get("reverse", False))
        if reverse != self._reverse:
            self._reverse = reverse
            self.dir_switch.setChecked(reverse)

        self._recent = list(snap.get("recent") or ())
        configurable = bool(snap.get("configurable", True))
        if configurable != self._configurable:
            self._configurable = configurable
            self.sheet_btn.setEnabled(configurable)
            self.recent_btn.setEnabled(configurable)
            self.out_btn.setEnabled(configurable)
            self.dir_switch.setEnabled(configurable)
            self.format_switch.setEnabled(configurable)
            for widget in self.ups_boxes + self.ups_combos:
                widget.setEnabled(configurable)

        self._sheet_dir = snap.get("sheet_dir", "")
        self._out_dir = snap.get("labeldir", "")
        self._capture_dir = snap.get("capturedir", "")
        meta = (snap.get("sheet", ""), self._out_dir, self._capture_dir)
        if meta != self._meta:
            self._meta = meta
            self.sheet_label.setText(meta[0])
            self.sheet_label.setToolTip(os.path.join(self._sheet_dir, meta[0]))
            for label, folder in ((self.out_label, meta[1]),
                                  (self.capture_label, meta[2])):
                full = os.path.abspath(folder or ".")
                label.setText(_wrap_path(full))
                label.setToolTip(full)

        self._show_fault(snap)

        note = snap.get("note") or "Press START to read the labels and run."
        if self.status.text() != note:
            self.status.setText(note)
            colour = BAD if state in ("rewind", "mismatch", "unread",
                                      "incomplete") else (
                WARN if state == "reading" else MUTED)
            self.status.setStyleSheet(f"color: {colour};")

        debug = bool(snap.get("debug"))
        if debug != self._debug:
            self._debug = debug
            self.debug_box.setVisible(debug)
            self.dock.setVisible(debug)
        if debug:
            self.debug_label.setText(
                f"{snap.get('fps', 0):.1f} fps\n{snap.get('dets', 0)} "
                f"detections\n{snap.get('status', '')}")
            view = snap.get("window_view")
            if view is not None:
                self.view_label.setPixmap(to_pixmap(view))
