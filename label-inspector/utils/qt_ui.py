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

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt

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
    "idle": ("IDLE", MUTED),
}


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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._pixmap = None
        self._snap = {}
        self._caption_font = QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Bold)
        self._banner_font = QtGui.QFont("DejaVu Sans", 15, QtGui.QFont.Bold)
        self._line_font = QtGui.QFont("DejaVu Sans", 10)

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
        self._draw_boxes(p)
        self._draw_tags(p)
        self._draw_fault(p)
        p.restore()

    def _text_rect(self, p, font, text, x, y, pad=4):
        p.setFont(font)
        m = QtGui.QFontMetrics(font)
        w = m.horizontalAdvance(text) * self._inv + 2 * pad
        h = m.height() * self._inv + pad
        return QtCore.QRectF(x, y, w, h)

    def _scaled(self, font):
        """A font that comes out the requested size on screen whatever the
        picture has been scaled to."""
        f = QtGui.QFont(font)
        f.setPointSizeF(max(font.pointSizeF() * self._inv, 1.0))
        return f

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
        font = self._scaled(QtGui.QFont("DejaVu Sans", 10, QtGui.QFont.Bold))
        m = QtGui.QFontMetrics(font)
        for x1, y1, x2, y2, rgb, caption in self._snap.get("tags", ()):
            colour = QtGui.QColor(*rgb)
            p.setPen(QtGui.QPen(colour, 4 * self._inv))
            p.setBrush(Qt.NoBrush)
            p.drawRect(QtCore.QRectF(x1 - 3 * self._inv, y1 - 3 * self._inv,
                                     x2 - x1 + 6 * self._inv,
                                     y2 - y1 + 6 * self._inv))
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
        """A red border round the picture, the headline in a bar across the
        top, and the detail stacked at the bottom."""
        snap = self._snap
        banner = snap.get("banner")
        lines = snap.get("lines") or ()
        if not banner and not lines:
            return

        w = self._pixmap.width()
        h = self._pixmap.height()
        red = QtGui.QColor(BAD)
        p.setPen(QtGui.QPen(red, 6 * self._inv))
        p.setBrush(Qt.NoBrush)
        p.drawRect(QtCore.QRectF(2, 2, w - 4, h - 4))

        if banner:
            font = self._scaled(self._banner_font)
            m = QtGui.QFontMetrics(font)
            bar = QtCore.QRectF(0, 8 * self._inv, w,
                                m.height() + 14 * self._inv)
            p.setPen(Qt.NoPen)
            p.setBrush(red)
            p.drawRect(bar)
            p.setPen(QtGui.QColor("#000000"))
            p.setFont(font)
            p.drawText(bar, Qt.AlignCenter, banner)

        if not lines:
            return
        font = self._scaled(self._line_font)
        m = QtGui.QFontMetrics(font)
        pad = 6 * self._inv
        step = m.height() + pad
        y = h - 10 * self._inv - step * len(lines)
        for text, rgb in lines:
            width = m.horizontalAdvance(text) + 2 * pad
            box = QtCore.QRectF(10 * self._inv, y, width, step)
            p.setPen(Qt.NoPen)
            p.setBrush(QtGui.QColor(0, 0, 0, 210))
            p.drawRect(box)
            p.setPen(QtGui.QColor(*rgb))
            p.setFont(font)
            p.drawText(box.adjusted(pad, 0, 0, 0),
                       Qt.AlignVCenter | Qt.AlignLeft, text)
            y += step


class InspectorWindow(QtWidgets.QMainWindow):
    """The console. Emits `command(name, argument)` and nothing else."""

    command = QtCore.pyqtSignal(str, object)
    _incoming = QtCore.pyqtSignal(object)

    KEYS = {Qt.Key_S: "start", Qt.Key_X: "stop", Qt.Key_O: "sheet",
            Qt.Key_F: "outdir", Qt.Key_D: "debug", Qt.Key_Q: "quit"}

    def __init__(self, title="LABEL INSPECTOR"):
        super().__init__()
        self.setWindowTitle("Label Inspector")
        self.resize(1180, 900)
        self._debug = False
        self._configurable = True
        self._state = "idle"
        self._meta = None
        self._sheet_dir = ""
        self._out_dir = ""
        self._build(title)
        # Queued by default across threads, so the worker can post a snapshot
        # without touching a widget.
        self._incoming.connect(self._apply, Qt.QueuedConnection)

    # -- construction -----------------------------------------------------
    def _build(self, title):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {INK}; color: {TEXT};
                                    font-family: 'DejaVu Sans'; }}
            QLabel#title {{ font-size: 21px; font-weight: 600;
                            letter-spacing: 1px; }}
            QLabel#meta  {{ color: {MUTED}; font-size: 11px; }}
            QLabel#pill  {{ font-size: 13px; font-weight: 700;
                            padding: 6px 18px; border-radius: 13px; }}
            QLabel#status{{ color: {MUTED}; font-size: 12px; }}
            QLabel#hint  {{ color: #6d757d; font-size: 11px; }}
            QFrame#rule  {{ background: {LINE}; max-height: 1px;
                            border: none; }}
            QFrame#header, QFrame#footer {{ background: {INK}; }}
            QPushButton {{ font-size: 15px; font-weight: 600;
                           border-radius: 6px; padding: 14px 8px;
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
            QPushButton#secondary {{ font-size: 12px; font-weight: 600;
                                     color: {ACCENT}; padding: 10px 8px; }}
            QGroupBox {{ border: 1px solid {LINE}; border-radius: 6px;
                         margin-top: 10px; font-size: 11px; color: {MUTED}; }}
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
        self.video = VideoView()
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

        stack = QtWidgets.QVBoxLayout()
        stack.setSpacing(2)
        self.title_label = QtWidgets.QLabel(title, objectName="title")
        self.meta_label = QtWidgets.QLabel("", objectName="meta")
        stack.addWidget(self.title_label)
        stack.addWidget(self.meta_label)
        row.addLayout(stack)
        row.addStretch(1)

        self.pill = QtWidgets.QLabel("IDLE", objectName="pill")
        self._set_pill("idle")
        row.addWidget(self.pill)
        return bar

    def _sidebar(self):
        side = QtWidgets.QWidget()
        side.setFixedWidth(230)
        col = QtWidgets.QVBoxLayout(side)
        col.setContentsMargins(14, 14, 14, 14)
        col.setSpacing(8)

        self.start_btn = QtWidgets.QPushButton("START", objectName="start")
        self.stop_btn = QtWidgets.QPushButton("STOP", objectName="stop")
        self.sheet_btn = QtWidgets.QPushButton("LOAD SHEET",
                                               objectName="secondary")
        self.out_btn = QtWidgets.QPushButton("OUTPUT FOLDER",
                                             objectName="secondary")
        self.start_btn.clicked.connect(lambda: self.command.emit("start", None))
        self.stop_btn.clicked.connect(lambda: self.command.emit("stop", None))
        self.sheet_btn.clicked.connect(self._choose_sheet)
        self.out_btn.clicked.connect(self._choose_output)
        for b in (self.start_btn, self.stop_btn):
            b.setFixedHeight(58)
        col.addWidget(self.start_btn)
        col.addWidget(self.stop_btn)
        col.addSpacing(6)
        col.addWidget(self._rule())
        col.addSpacing(6)
        col.addWidget(self.sheet_btn)
        col.addWidget(self.out_btn)

        self.debug_box = QtWidgets.QGroupBox("DIAGNOSTICS")
        inner = QtWidgets.QVBoxLayout(self.debug_box)
        inner.setContentsMargins(10, 6, 10, 10)
        self.debug_label = QtWidgets.QLabel("", objectName="meta")
        self.debug_label.setWordWrap(True)
        inner.addWidget(self.debug_label)
        col.addSpacing(10)
        col.addWidget(self.debug_box)
        self.debug_box.hide()

        col.addStretch(1)
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
        row.addWidget(QtWidgets.QLabel(
            "[S] START   [X] STOP   [O] SHEET   [F] FOLDER   "
            "[D] DIAGNOSTICS   [Q] QUIT", objectName="hint"))
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
            self, "Choose the folder to save results in", self._out_dir)
        if path:
            self.command.emit("outdir", path)

    def keyPressEvent(self, event):
        name = self.KEYS.get(event.key())
        if name is None:
            return super().keyPressEvent(event)
        if name == "sheet":
            if self._configurable:
                self._choose_sheet()
        elif name == "outdir":
            if self._configurable:
                self._choose_output()
        else:
            self.command.emit(name, None)

    def closeEvent(self, event):
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
        if state == "idle":
            self.pill.setStyleSheet(f"color: {MUTED}; background: {PANEL}; "
                                    f"border: 1px solid {LINE};")
        else:
            self.pill.setStyleSheet(f"color: {INK}; background: {colour};")

    def _apply(self, snap):
        self.video.set_snapshot(snap)

        state = snap.get("state", "idle")
        if state != self._state:
            self._state = state
            self._set_pill(state)
            running = state in ("running", "reading")
            self.start_btn.setEnabled(state in ("idle", "rewind"))
            self.stop_btn.setEnabled(running)

        configurable = bool(snap.get("configurable", True))
        if configurable != self._configurable:
            self._configurable = configurable
            self.sheet_btn.setEnabled(configurable)
            self.out_btn.setEnabled(configurable)

        self._sheet_dir = snap.get("sheet_dir", "")
        self._out_dir = snap.get("outdir", "")
        meta = f"SHEET  {snap.get('sheet', '')}     OUT  {self._out_dir}"
        if meta != self._meta:
            # An output path can be longer than the header, and the sheet
            # name is the half that identifies the run, so the middle of the
            # path is what gives way.
            self._meta = meta
            room = max(self.meta_label.width(), 320)
            m = QtGui.QFontMetrics(self.meta_label.font())
            self.meta_label.setText(m.elidedText(meta, Qt.ElideMiddle, room))
            self.meta_label.setToolTip(meta)

        note = snap.get("note") or "Press START to read the labels and run."
        if self.status.text() != note:
            self.status.setText(note)
            colour = BAD if state == "rewind" else (
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
