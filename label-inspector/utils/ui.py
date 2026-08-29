"""A minimal on-frame control panel for the imshow window.

OpenCV's own cv2.createButton needs the Qt highgui backend; this build is
GTK3, so the buttons are drawn onto the frame and clicks are hit-tested
against their rectangles in a mouse callback. GTK reports mouse positions in
image coordinates, so a resized WINDOW_NORMAL still lands its clicks in the
right place.
"""

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

GREEN = (60, 200, 60)
RED   = (60, 60, 220)
GREY  = (90, 90, 90)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Button:
    """A labelled rectangle that runs `action` when clicked."""

    def __init__(self, label, rect, color, action, key=None):
        self.label = label
        self.rect = rect            # (x, y, w, h)
        self.color = color
        self.action = action
        self.key = key              # keyboard equivalent, shown on the button
        self.enabled = True

    def contains(self, x, y):
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def draw(self, frame, scale):
        bx, by, bw, bh = self.rect
        color = self.color if self.enabled else GREY
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), WHITE, 2)

        (tw, th), _ = cv2.getTextSize(self.label, FONT, scale, 2)
        cv2.putText(frame, self.label,
                    (bx + (bw - tw) // 2, by + (bh + th) // 2),
                    FONT, scale, WHITE if self.enabled else (200, 200, 200),
                    2, cv2.LINE_AA)
        if self.key:
            cv2.putText(frame, f"[{self.key}]", (bx + 8, by + bh - 8),
                        FONT, scale * 0.45, WHITE, 1, cv2.LINE_AA)


class ControlPanel:
    """START / STOP for the winding machine, plus a machine-state readout.

    Sizes itself from the frame so it stays legible whatever resolution the
    camera runs at. `note` carries why the machine stopped, so an operator
    looking at the window can see a validation failure caused it.
    """

    def __init__(self, width, height, on_start, on_stop):
        self.width = width
        self.height = height
        self.scale = max(width / 1400.0, 0.6)

        bw = int(width * 0.20)
        bh = int(bw * 0.38)
        pad = int(width * 0.02)
        x = width - bw - pad
        top = pad

        self.buttons = [
            Button("START", (x, top, bw, bh), GREEN, on_start, key="s"),
            Button("STOP", (x, top + bh + pad // 2, bw, bh), RED, on_stop, key="x"),
        ]
        self.state_at = (x, top + 2 * (bh + pad // 2) + int(bh * 0.5))
        self.note = None

    # ── input ─────────────────────────────────────────────────────────────
    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.enabled and button.contains(x, y):
                button.action()
                return

    def on_key(self, key):
        """Keyboard equivalents, so the panel works without a mouse.
        Returns True if the key was one of ours."""
        if key < 0 or key > 255:
            return False
        char = chr(key).lower()
        for button in self.buttons:
            if button.key == char and button.enabled:
                button.action()
                return True
        return False

    # ── output ────────────────────────────────────────────────────────────
    def draw(self, frame, running):
        start, stop = self.buttons
        start.enabled = not running
        stop.enabled = running
        for button in self.buttons:
            button.draw(frame, self.scale)

        x, y = self.state_at
        text = "MACHINE: RUNNING" if running else "MACHINE: STOPPED"
        cv2.putText(frame, text, (x, y), FONT, self.scale * 0.75,
                    GREEN if running else RED, 2, cv2.LINE_AA)
        if not running and self.note:
            cv2.putText(frame, self.note, (x, y + int(34 * self.scale)),
                        FONT, self.scale * 0.6, RED, 2, cv2.LINE_AA)
        return frame
