"""The operator console drawn onto the camera frame.

OpenCV's own cv2.createButton needs the Qt highgui backend; this build is
GTK3, so the whole console is drawn onto the frame and clicks are hit-tested
against their rectangles in a mouse callback. GTK reports mouse positions in
image coordinates, so a resized WINDOW_NORMAL still lands its clicks in the
right place.

Layout is a header band across the top, a column of controls down the right,
and a status bar across the bottom. Everything is sized from the frame, so the
same code lays out sensibly whether the camera is running at 1280x720 or at
the full 1944x2592 the global shutter gives after rotation. The app reserves
the bands it draws in — `header_h` and `footer_h` — so overlays that write on
the picture itself can keep clear of them.

Every string drawn here is ASCII. Hershey, the only font OpenCV ships, has no
glyph for anything else: an em dash or an ellipsis comes out as '???'.
"""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_HEAVY = cv2.FONT_HERSHEY_DUPLEX

# BGR. A dark, low-chroma console so the picture stays the brightest thing on
# screen and the chrome reads as chrome.
INK      = (26, 22, 19)          # header / footer ground
PANEL    = (46, 40, 35)          # raised surface
LINE     = (78, 70, 62)          # hairline divider
TEXT     = (240, 238, 236)
MUTED    = (170, 162, 154)
ACCENT   = (237, 145, 55)        # steel blue
OK       = (96, 176, 64)
WARN     = (74, 196, 240)
BAD      = (84, 84, 232)
DISABLED = (92, 86, 80)


def rounded_rect(img, rect, radius, colour, thickness=-1):
    """A filled or outlined rectangle with round corners.

    OpenCV has no primitive for this, and square corners are most of what
    makes a drawn-on panel look like a debug overlay rather than a console.
    """
    x, y, w, h = rect
    r = int(max(0, min(radius, w // 2, h // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), colour, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), colour, -1)
        for cx, cy in ((x + r, y + r), (x + w - r, y + r),
                       (x + r, y + h - r), (x + w - r, y + h - r)):
            cv2.circle(img, (cx, cy), r, colour, -1)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), colour, thickness)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), colour, thickness)
    cv2.line(img, (x, y + r), (x, y + h - r), colour, thickness)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), colour, thickness)
    for (cx, cy), a in (((x + r, y + r), 180), ((x + w - r, y + r), 270),
                        ((x + w - r, y + h - r), 0), ((x + r, y + h - r), 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, colour, thickness)


# Hershey has no glyph outside ASCII, and OpenCV draws '???' for anything it
# cannot render. Notes reach the panel straight from log messages, which are
# written with proper dashes, so every string is folded on the way in rather
# than every caller being asked to remember.
_SUBS = {"\u2014": "-", "\u2013": "-", "\u2026": "...", "\u2192": "->",
         "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
         "\u00d7": "x", "\u2022": "*", "\u00b7": "-"}


def ascii_only(s):
    s = "".join(_SUBS.get(ch, ch) for ch in str(s if s is not None else ""))
    return s.encode("ascii", "replace").decode("ascii")


def text_at(img, s, x, y, colour, scale, thick=2, font=FONT):
    cv2.putText(img, ascii_only(s), (int(x), int(y)), font, scale, colour,
                thick, cv2.LINE_AA)


def text_w(s, scale, thick=2, font=FONT):
    return cv2.getTextSize(ascii_only(s), font, scale, thick)[0][0]


def ellipsize(s, limit, scale, thick=2, font=FONT):
    """Trim from the left, keeping the tail: for a path or a payload the end
    is the part that identifies it."""
    s = s or ""
    if text_w(s, scale, thick, font) <= limit:
        return s
    while s and text_w(".." + s, scale, thick, font) > limit:
        s = s[1:]
    return ".." + s


class Button:
    """A labelled rectangle that runs `action` when clicked.

    `primary` buttons are filled in their colour and carry the two actions an
    operator uses constantly; the rest are outlined, which is what keeps the
    eye on START and STOP.
    """

    def __init__(self, label, action, colour, key=None, primary=True,
                 hint=None):
        self.label = label
        self.hint = hint or label      # short form, for the status bar
        self.action = action
        self.colour = colour
        self.key = key
        self.primary = primary
        self.enabled = True
        self.rect = (0, 0, 0, 0)

    def contains(self, x, y):
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def draw(self, img, scale, text_scale=None):
        bx, by, bw, bh = self.rect
        radius = int(bh * 0.22)
        colour = self.colour if self.enabled else DISABLED
        label_colour = TEXT if self.enabled else MUTED

        if self.primary:
            rounded_rect(img, self.rect, radius, colour, -1)
        else:
            rounded_rect(img, self.rect, radius, PANEL, -1)
            rounded_rect(img, self.rect, radius, colour if self.enabled
                         else LINE, 2)
            label_colour = colour if self.enabled else MUTED

        # No key badge on the button itself: the status bar lists every
        # shortcut in one place, and a stray letter in the corner of a button
        # is the sort of thing that makes a console look unfinished.
        s = (text_scale if text_scale is not None else scale)
        s *= 0.95 if self.primary else 0.72
        # Type is sized for the screen, the button for the frame, so on a
        # heavily shrunk display the label can outgrow its button. Give it
        # back whatever it needs rather than letting it run over the edge.
        room = bw - int(bh * 0.36)
        tw = text_w(self.label, s, 2)
        if tw > room:
            s *= room / float(tw)
            tw = text_w(self.label, s, 2)
        text_at(img, self.label, bx + (bw - tw) // 2,
                by + (bh + int(24 * s)) // 2, label_colour, s, 2)


class ControlPanel:
    """START / STOP, the run's configuration, and a machine-state readout.

    The app sets `sheet_name`, `output_dir` and `note` on the panel; the panel
    owns nothing but its own layout. `configurable` gates the two buttons that
    change where data comes from and goes to — swapping either mid-run would
    leave half a run recorded against one sheet and half against another, so
    they are only live when the machine is idle.
    """

    STATES = {
        "running": ("RUNNING",  OK,    True),
        "reading": ("READING",  WARN,  True),
        "rewind":  ("REWIND",   BAD,   True),
        "mismatch": ("WRONG SHEET", BAD, True),
        "unread":  ("NOT READING", BAD, True),
        "incomplete": ("BAD LABEL", BAD, True),
        "idle":    ("IDLE",     MUTED, False),
    }

    def __init__(self, width, height, on_start, on_stop,
                 on_load_sheet=None, on_output_dir=None,
                 title="LABEL INSPECTOR", display_scale=1.0):
        self.width = width
        self.height = height
        self.title = title

        # Two scales, because the console is drawn at capture resolution and
        # then shrunk to fit the screen. Boxes can be sized against the frame
        # -- a button that is a fifth of the picture stays a fifth of it -- but
        # text cannot: a caption sized for a 1944px frame that is then shown
        # at 0.37 comes out at a third the height it was drawn, which is what
        # made the old console unreadable. So type is sized against the frame
        # as it will appear on screen, and grows as the shrink gets harsher.
        self.display_scale = max(float(display_scale or 1.0), 0.05)
        self.scale = max(width / 1500.0, 0.55)
        self.text = max(self.scale, 1.0 / self.display_scale)

        self.header_h = max(76, int(74 * self.text))
        self.footer_h = max(52, int(42 * self.text))

        self.sheet_name = ""
        self.output_dir = ""
        self.note = None
        self.configurable = True

        self.buttons = [
            Button("START", on_start, OK, key="s"),
            Button("STOP", on_stop, BAD, key="x"),
        ]
        self.config_buttons = []
        if on_load_sheet is not None:
            self.config_buttons.append(
                Button("LOAD SHEET", on_load_sheet, ACCENT, key="o",
                       primary=False, hint="SHEET"))
        if on_output_dir is not None:
            self.config_buttons.append(
                Button("LABEL FOLDER", on_output_dir, ACCENT, key="f",
                       primary=False, hint="FOLDER"))
        self.buttons += self.config_buttons
        self._layout()

    def _layout(self):
        s = self.scale
        bw = int(self.width * 0.24)
        bh = int(bw * 0.30)
        pad = int(self.width * 0.018)
        x = self.width - bw - pad
        y = self.header_h + pad
        # Where the picture ends and the control column begins. Overlays that
        # write on the picture use this so they do not run under the buttons.
        self.content_right = x - pad

        start, stop = self.buttons[0], self.buttons[1]
        start.rect = (x, y, bw, bh)
        stop.rect = (x, y + bh + int(pad * 0.5), bw, bh)

        y = y + 2 * bh + int(pad * 1.9)
        self.rule_y = y - pad
        self.rule_x = (x, x + bw)
        small = int(bh * 0.72)
        for i, button in enumerate(self.config_buttons):
            button.rect = (x, y + i * (small + int(pad * 0.4)), bw, small)

    # -- input ------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.enabled and button.contains(x, y):
                button.action()
                return

    def on_key(self, key):
        """Keyboard equivalents, so the console works without a mouse.
        Returns True if the key was one of ours."""
        if key < 0 or key > 255:
            return False
        char = chr(key).lower()
        for button in self.buttons:
            if button.key == char and button.enabled:
                button.action()
                return True
        return False

    # -- output -----------------------------------------------------------
    def _draw_header(self, img, state):
        s = self.scale
        h = self.header_h
        cv2.rectangle(img, (0, 0), (self.width, h), INK, -1)
        cv2.line(img, (0, h), (self.width, h), LINE, 2)

        # An accent block instead of a logo: cheap, and it reads as a brand
        # mark rather than as one more line of text.
        bar_w = max(6, int(10 * s))
        pad = int(self.width * 0.018)
        cv2.rectangle(img, (pad, int(h * 0.24)),
                      (pad + bar_w, int(h * 0.76)), ACCENT, -1)

        t = self.text
        x = pad + bar_w + int(16 * s)
        text_at(img, self.title, x, int(h * 0.44), TEXT, t * 0.95, 2,
                FONT_HEAVY)

        # The sheet name is what identifies the run, so it is drawn whole and
        # only the output path is trimmed to whatever room is left.
        ms = t * 0.62
        y = int(h * 0.82)
        edge = self.content_right - int(20 * s)
        if self.sheet_name:
            label = f"SHEET  {self.sheet_name}"
            text_at(img, label, x, y, MUTED, ms, 2)
            x += text_w(label, ms, 2) + int(30 * t)
        if self.output_dir and edge - x > int(60 * t):
            room = edge - x - text_w("OUT  ", ms, 2)
            text_at(img, "OUT  " + ellipsize(self.output_dir, room, ms, 2),
                    x, y, MUTED, ms, 2)
        self._draw_pill(img, state)

    def _draw_pill(self, img, state):
        s = self.scale
        label, colour, filled = self.STATES.get(state, self.STATES["idle"])
        pad = int(self.width * 0.018)
        ts = self.text * 0.78
        tw = text_w(label, ts, 2)
        pw = tw + int(56 * self.text)
        ph = int(self.header_h * 0.50)
        px = self.width - pad - pw
        py = (self.header_h - ph) // 2
        rounded_rect(img, (px, py, pw, ph), ph // 2,
                     colour if filled else PANEL, -1)
        if not filled:
            rounded_rect(img, (px, py, pw, ph), ph // 2, LINE, 2)
        cv2.circle(img, (px + int(20 * self.text), py + ph // 2),
                   max(4, int(7 * self.text)), INK if filled else colour, -1)
        text_at(img, label, px + int(36 * self.text),
                py + (ph + int(20 * ts)) // 2,
                INK if filled else colour, ts, 2)

    def _draw_footer(self, img, state):
        s = self.scale
        top = self.height - self.footer_h
        cv2.rectangle(img, (0, top), (self.width, self.height), INK, -1)
        cv2.line(img, (0, top), (self.width, top), LINE, 2)

        if self.note:
            message, colour = self.note, WARN if state == "reading" else BAD
        elif state == "running":
            message, colour = "Validating against the sheet.", MUTED
        else:
            message, colour = "Press START to read the labels and run.", MUTED

        t = self.text
        hints = " ".join(f"[{b.key.upper()}]{b.hint}"
                         for b in self.buttons if b.key)
        hs, ms = t * 0.48, t * 0.62
        hint_w = text_w(hints, hs, 1)
        baseline = top + (self.footer_h + int(20 * ms)) // 2
        left = int(self.width * 0.018)
        room = self.width - hint_w - left - int(self.width * 0.03)
        # A status line that has lost its opening words is worse than one set
        # a little smaller, so it gives up size before it gives up words.
        wide = text_w(message, ms, 2)
        if wide > room:
            ms = max(ms * room / float(wide), t * 0.45)
        text_at(img, ellipsize(message, room, ms, 2), left, baseline,
                colour, ms, 2)
        text_at(img, hints, self.width - hint_w - int(self.width * 0.018),
                baseline, MUTED, hs, 1)

    def draw(self, frame, state="idle"):
        """`state` is one of running / reading / rewind / idle. A bare bool is
        accepted too, so an older caller still works."""
        if state is True:
            state = "running"
        elif state is False:
            state = "idle"

        running = state in ("running", "reading")
        self.buttons[0].enabled = state in ("idle", "rewind", "mismatch",
                                            "unread", "incomplete")
        self.buttons[1].enabled = running
        for button in self.config_buttons:
            button.enabled = self.configurable and not running

        self._draw_header(frame, state)
        for button in self.buttons:
            button.draw(frame, self.scale, self.text)
        if self.config_buttons:
            x1, x2 = self.rule_x
            cv2.line(frame, (x1, self.rule_y), (x2, self.rule_y), LINE, 2)
        self._draw_footer(frame, state)
        return frame
