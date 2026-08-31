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


def _short_path(path, keep=2):
    """The tail of a path, which is the part that says which run this is.

    An output folder can be a dozen components deep and the panel is 320px
    wide; the full path is on the tooltip for anyone who wants it.
    """
    path = (path or "").rstrip(os.sep)
    if not path:
        return ""
    parts = path.split(os.sep)
    if len(parts) <= keep:
        return path
    return ".." + os.sep + os.sep.join(parts[-keep:])


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


class InspectorWindow(QtWidgets.QMainWindow):
    """The console. Emits `command(name, argument)` and nothing else."""

    command = QtCore.pyqtSignal(str, object)
    _incoming = QtCore.pyqtSignal(object)

    # Only the diagnostics toggle. Everything that moves the machine or
    # repoints the run is a button you have to look at and click: a stray
    # keypress on a console standing next to a winder must not be able to
    # stop the line, and must not be able to pop a file dialog over it
    # either. F11 is here because it is how you get out of full screen.
    KEYS = {Qt.Key_D: "debug", Qt.Key_F11: "fullscreen"}

    def __init__(self, title="VIKBOT Label Inspect"):
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
        # with 11px captions, held between sane limits either way.
        self._k = max(0.85, min(area.height() / 900.0, 2.2))
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
        self._running = False
        self._fault_text = None
        # Set False to let the window close on a single click, the way an
        # ordinary application does.
        self.locked = True
        self._build(title)
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

        self.fault_head = QtWidgets.QLabel("", objectName="faulthead")
        self.fault_head.setWordWrap(True)
        self.fault_body = QtWidgets.QLabel("", objectName="faultbody")
        self.fault_body.setWordWrap(True)
        self.fault_body.setTextFormat(Qt.RichText)
        self.fault_body.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        col.addWidget(self.fault_head)
        col.addWidget(self.fault_body)
        col.addStretch(1)
        self.left.hide()
        return self.left

    def _sidebar(self):
        side = QtWidgets.QWidget()
        side.setFixedWidth(self._side_w)
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
            b.setFixedHeight(int(58 * self._k))
        col.addWidget(self.start_btn)
        col.addWidget(self.stop_btn)
        col.addSpacing(6)
        col.addWidget(self._rule())
        col.addSpacing(6)
        col.addWidget(self.sheet_btn)
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
        run_col.addWidget(QtWidgets.QLabel("OUTPUT FOLDER",
                                           objectName="caption"))
        self.out_label = QtWidgets.QLabel("", objectName="value")
        self.out_label.setWordWrap(True)
        run_col.addWidget(self.out_label)
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
            "[D] DIAGNOSTICS   [F11] FULL SCREEN", objectName="hint"))
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
            self.command.emit(name, None)

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
        if state == "idle":
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
            if self.left.isVisible():
                self.left.hide()
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
        if not self.left.isVisible():
            self.left.show()

    def _apply(self, snap):
        self.video.set_snapshot(snap)

        state = snap.get("state", "idle")
        self._running = state in ("running", "reading")
        if state != self._state:
            self._state = state
            self._set_pill(state)
            self.start_btn.setEnabled(state in ("idle", "rewind"))
            self.stop_btn.setEnabled(self._running)

        configurable = bool(snap.get("configurable", True))
        if configurable != self._configurable:
            self._configurable = configurable
            self.sheet_btn.setEnabled(configurable)
            self.out_btn.setEnabled(configurable)

        self._sheet_dir = snap.get("sheet_dir", "")
        self._out_dir = snap.get("outdir", "")
        meta = (snap.get("sheet", ""), self._out_dir)
        if meta != self._meta:
            self._meta = meta
            self.sheet_label.setText(meta[0])
            self.sheet_label.setToolTip(os.path.join(self._sheet_dir, meta[0]))
            self.out_label.setText(_short_path(meta[1]))
            self.out_label.setToolTip(os.path.abspath(meta[1] or "."))

        self._show_fault(snap)

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
