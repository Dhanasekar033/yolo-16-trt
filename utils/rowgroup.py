"""Group every label in the frame into physical rows, and judge each row as
a unit against one spreadsheet row.

This is the middle ground between the two modes that came before it.

`CenterLineQRDecoder` judges a column of labels as it crosses a trigger line.
That gives a real row-wise verdict — the labels are read together, in their
on-screen order, so a swap between two positions is visible — but it only ever
looks at a narrow zone, so a column that crosses faster than a few frames is
missed outright, and the zone has to be tuned to the web's speed.

The rolling window looks at the whole frame and never misses a label, but it
matches each payload to the sheet on its own. Order stops mattering, and with
it goes the ability to say that four correct codes are in the wrong places.

What is actually wanted is the first one's verdict with the second one's
field of view: cluster the labels by x-centre wherever they are in frame,
follow each cluster as the web carries it across, and judge it as soon as it
has read everything it is going to — or when it leaves the frame short.

Everything needed for that already exists on `CenterLineQRDecoder`; the only
thing tying it to the trigger line is which labels it will look at. So this
is that class with the zone opened up to the whole frame, and with the
zone-tuning advice replaced by advice that makes sense frame-wide.
"""

from utils.qr import LABEL, CenterLineQRDecoder


class RowGroupDecoder(CenterLineQRDecoder):
    """Read every label in the frame, grouped into rows, judged in sequence.

    Same callbacks as `CenterLineQRDecoder`: `on_batch(texts)` fires once per
    physical row with the payloads in top-to-bottom slot order and None where
    a label never read, which is exactly what `SequenceValidator.check_batch`
    takes. `identify` resolves a payload to its sheet row so two records of
    the same group merge instead of being judged twice.
    """

    def __init__(self, label_cls, qr_cls, margin=0.15, min_px=8,
                 on_decode=None, on_batch=None, on_label=None, dump_dir=None,
                 column_gap=None, expect=None, source=LABEL, identify=None,
                 memory=900, min_dwell=3, check=None):
        # line_x/half_width/zone only feed the overlay and the zone filter,
        # both of which are replaced below. They are still passed through so
        # every inherited method that reads them finds a sane number.
        super().__init__(line_x=0, label_cls=label_cls, qr_cls=qr_cls,
                         margin=margin, min_px=min_px, on_decode=on_decode,
                         on_batch=on_batch, on_label=on_label,
                         dump_dir=dump_dir, half_width=0, zone=0,
                         column_gap=column_gap, expect=expect, source=source,
                         identify=identify, memory=memory,
                         min_dwell=min_dwell, check=check)

    # ── the whole frame is the zone ───────────────────────────────────────
    def _zone_labels(self, dets):
        """Every label in the frame, not just the ones near a line.

        This one override is what turns the trigger-line reader into a
        frame-wide one. Tracking, slot assignment, identity merging and the
        judge-when-complete rule above it are all already frame-general: a
        group is followed from wherever it enters, judged the moment it has
        read every position being checked, and judged short if it reaches the
        far edge without them.
        """
        return [d for d in dets if int(d[5]) == self.label_cls]

    def update(self, frame, dets):
        # `zone` is the half-width a group is readable across, and frame-wide
        # that is half the frame. Keeping it current makes `frames_in_zone`
        # mean "frames a label gets while it is on screen", which is the
        # number the dwell warning below is about.
        if hasattr(frame, "shape"):
            self.zone = frame.shape[1] / 2.0
        return super().update(frame, dets)

    # ── advice ────────────────────────────────────────────────────────────
    def _check_dwell(self, column):
        """Warn when the web outruns the frame rate.

        The inherited version offers a wider --zone as the fix. Frame-wide
        there is no zone left to widen — if a row crosses the whole frame in
        too few frames, the only answers are more frames or a slower web — so
        the advice is rewritten and the rest of the bookkeeping kept.
        """
        self.dwells.append(column.last_frame - column.first_frame + 1)
        del self.dwells[:-10]

        if not self.settled:
            self.unsettled_run += 1
            if self.unsettled_run in (8, 40) or self.unsettled_run % 400 == 0:
                print("[row] cannot lock onto the web's motion — rows are "
                      "crossing faster than the frame rate can follow, so "
                      "some will be missed entirely. Raise FPS (smaller "
                      "--imgsz or capture size) or slow the web.")
            return
        self.unsettled_run = 0

        available = self.frames_in_zone
        if not available or available >= self.min_dwell:
            self.thin_run = 0
            return
        self.thin_run += 1
        if self.thin_run not in (3, 30) and self.thin_run % 300:
            return
        print(f"[row] a row is only on screen for {available:.1f} frame(s) at "
              f"{self.step:.0f} px/frame — too few decode attempts per label. "
              f"Raise FPS (smaller --imgsz or capture size) or slow the web.")

    # ── overlay ───────────────────────────────────────────────────────────
    def draw(self, frame):
        """Boxes round what has been read, and a running count.

        Deliberately not the inherited overlay: that one is built around a
        trigger line and a zone band, neither of which exists here.
        """
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        for box, text in self.results:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0) if text else (0, 165, 255), 2)

        groups = len(self.tracked)
        waiting = sum(1 for c in self.tracked if not c.judged)
        cv2.putText(frame, f"ROWS in frame: {groups}  unjudged: {waiting}  "
                           f"codes read: {self.count}",
                    (20, 76), font, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

        head = self.head
        if head is not None:
            marks = " ".join(f"D{i + 1}:{'OK' if i in head.reads else '..'}"
                             for i in range(len(head.slots_y)))
            cv2.putText(frame, f"reading  {marks}", (20, 108), font, 0.7,
                        (255, 0, 255), 2, cv2.LINE_AA)
        return frame
