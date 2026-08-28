"""A side window showing the sheet against what the camera actually read.

One table per crossing: the position on screen, the payload the spreadsheet
expected there, the payload that was decoded, and the verdict — so a fault can
be read off directly instead of reconstructed from console lines.

The `labels seen` line at the top is the one to watch. Positions are assigned
top to bottom, so a column that arrives with fewer labels than the sheet has
columns shifts every label below the missing one into the wrong position, and
the table then shows a row of SWAPPED that has nothing to do with the labels
themselves being out of order.
"""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

BG      = (32, 32, 32)
GRID    = (70, 70, 70)
HEAD    = (200, 200, 200)
DIM     = (130, 130, 130)
WHITE   = (240, 240, 240)
GREEN   = (80, 220, 80)
RED     = (70, 70, 235)
AMBER   = (60, 200, 235)
MAGENTA = (220, 80, 220)

STATUS_COLOR = {
    "OK":        GREEN,
    "SKIPPED":   DIM,
    "NO-READ":   AMBER,
    "SWAPPED":   MAGENTA,
    "WRONG-ROW": RED,
    "UNKNOWN":   RED,
}


def tail(text, n=24):
    """Payloads share a long prefix; the tail is the part that differs.

    ASCII only — cv2's Hershey fonts have no glyph for an ellipsis and draw
    it as question marks."""
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= n else ".." + text[-(n - 2):]


class MappingView:
    """Renders the expected-vs-decoded table for the most recent crossing."""

    WIDTH = 1060
    ROW_H = 34

    def __init__(self, per_row=4, history=18):
        self.per_row = per_row
        self.history = []          # [(row_number, ok)] most recent last
        self.history_max = history
        self.result = None
        self.stopped_by = None

    def update(self, result, stopped_by=None):
        """Take the latest verdict. `stopped_by` is the reason string when this
        crossing stopped the machine, or None."""
        if result is None:
            return
        self.result = result
        self.stopped_by = stopped_by
        self.history.append((result.row, result.ok))
        del self.history[:-self.history_max]

    # ── drawing helpers ───────────────────────────────────────────────────
    def _text(self, img, s, x, y, color=WHITE, scale=0.6, thick=1):
        cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    def render(self):
        rows = self.per_row
        height = 150 + rows * self.ROW_H + 70
        img = np.full((height, self.WIDTH, 3), BG, np.uint8)

        r = self.result
        if r is None:
            self._text(img, "waiting for the first crossing...", 24, 60, DIM, 0.7)
            return img

        # ── header: which sheet row, and the verdict ──────────────────────
        verdict = "PASS" if r.ok else "FAIL"
        vcolor = GREEN if r.ok else RED
        self._text(img, f"SHEET ROW {r.row}", 24, 42, WHITE, 0.9, 2)
        (tw, _), _ = cv2.getTextSize(verdict, FONT, 0.9, 2)
        self._text(img, verdict, self.WIDTH - tw - 24, 42, vcolor, 0.9, 2)
        if not r.ok:
            self._text(img, r.summary(), 24, 70, vcolor, 0.6)

        # ── the line that explains most faults ────────────────────────────
        seen = len(r.entries)
        short = seen != rows
        self._text(img, f"labels seen: {seen} of {rows}"
                        + ("   <- positions below the missing one are shifted"
                           if short else ""),
                   24, 98, AMBER if short else DIM, 0.6)

        # ── table ─────────────────────────────────────────────────────────
        top = 118
        cols = (24, 96, 400, 740)
        for label, x in zip(("POS", "EXPECTED (xlsx)", "DECODED (camera)",
                             "RESULT"), cols):
            self._text(img, label, x, top + 20, HEAD, 0.55)
        cv2.line(img, (16, top + 30), (self.WIDTH - 16, top + 30), GRID, 1)

        for i in range(rows):
            y = top + 30 + (i + 1) * self.ROW_H
            entry = next((e for e in r.entries if e.pos == i), None)
            color = STATUS_COLOR.get(entry.status, WHITE) if entry else DIM

            self._text(img, f"D{i + 1}", cols[0], y, HEAD, 0.6)
            if entry is None:
                self._text(img, "(no label at this position)", cols[1], y,
                           AMBER, 0.55)
                self._text(img, "MISSING", cols[3], y, AMBER, 0.55)
                continue

            self._text(img, tail(entry.expected) or "-", cols[1], y, DIM, 0.55)

            if entry.status == "SKIPPED":
                got = "(not checked)"
            elif entry.text:
                got = tail(entry.text)
            else:
                got = "(no read)"
            self._text(img, got, cols[2], y, color, 0.55)

            note = entry.status
            if entry.status == "SWAPPED" and entry.found:
                note = f"SWAPPED <- D{entry.found[1]}"
            elif entry.status == "WRONG-ROW" and entry.found:
                note = f"ROW {entry.found[0]} D{entry.found[1]}"
            self._text(img, note, cols[3], y, color, 0.55)

        # ── recent history strip ──────────────────────────────────────────
        y = top + 30 + (rows + 1) * self.ROW_H + 24
        self._text(img, "recent:", 24, y, HEAD, 0.55)
        x = 110
        for row_no, ok in self.history[-self.history_max:]:
            cv2.rectangle(img, (x, y - 14), (x + 44, y + 6),
                          GREEN if ok else RED, -1)
            self._text(img, str(row_no)[-4:], x + 4, y, (20, 20, 20), 0.45)
            x += 50
        return img
