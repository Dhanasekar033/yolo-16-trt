"""Validate decoded QR payloads against the expected grid in an Excel sheet.

The sheet (validation.xlsx) holds one physical row of labels per spreadsheet
row, with the payloads in the `QR DATA1..QR DATA4` columns:

    QR DATA1                          QR DATA2            QR DATA3   QR DATA4
    HTTPS://SCAN.SMARTQR.IO/LS5/7016  HTTPS://...ZNR2...  ...        ...
    HTTPS://SCAN.SMARTQR.IO/LS5/7015  HTTPS://...ZJAW...  ...        ...

Every payload in the sheet is unique, and that is what the checking leans on:
a decoded code identifies *itself* — which sheet row and which QR DATA column
it belongs to — so nothing depends on the order the codes happened to decode
in. Labels sitting at a slight angle cross the trigger line bottom-first, and
a reader that trusted decode order would call that a mismatch; looking the
payload up cannot be fooled by it.

A crossing is judged on three things, each reported separately:

  * identity — is every code one the sheet knows at all?
  * placement — is each code in the QR DATA column it belongs to, reading the
    labels top to bottom on screen?
  * sequence — is this the sheet row that was expected to come next?
"""

import re
from collections import Counter

import cv2
import openpyxl

QR_COL_RE = re.compile(r"^\s*QR\s*DATA\s*(\d+)\s*$", re.IGNORECASE)

# Written by utils/prepare.py onto the working copy of a sheet: which symbol
# a row's codes are printed as, which of the datamatrix's repeated printings
# this row is, and the row of the operator's own sheet it came from. A sheet
# that has not been through the expansion has none of these, and every row of
# it is read as an ordinary row of QR codes.
TYPE_COL   = "CODE TYPE"
PRINT_COL  = "PRINT NO"
SOURCE_COL = "SOURCE ROW"

QR_CODE     = "qr"        # kind: the codes on this row are QR
DATA_MATRIX = "dm"        # kind: they are datamatrix, printed several times

OK        = "OK"
SWAPPED   = "SWAPPED"     # right row, but sitting in another column
WRONG_ROW = "WRONG-ROW"   # a real code, but from a different sheet row
UNKNOWN   = "UNKNOWN"     # decoded fine, but no such code in the sheet
NODECODE  = "NO-READ"     # a label was there, but it never read
SKIPPED   = "SKIPPED"     # this position is switched off, nothing checked

TOP_DOWN  = "top-down"
BOTTOM_UP = "bottom-up"


def normalize(text):
    """Compare payloads case- and whitespace-insensitively."""
    return (text or "").strip().upper()


class SheetRow:
    """One spreadsheet row: the QR DATA1..N payloads expected side by side."""

    __slots__ = ("index", "number", "texts", "kind", "printing", "source",
                 "group")

    def __init__(self, index, number, texts, kind=QR_CODE, printing=None,
                 source=None, group=None):
        self.index = index        # 0-based position in the sequence
        self.number = number      # 1-based spreadsheet row number
        self.texts = texts        # [QR DATA1, QR DATA2, ...]
        self.kind = kind          # QR_CODE or DATA_MATRIX
        self.printing = printing  # 1..N for a datamatrix row, else None
        self.source = source      # row of the operator's sheet it came from
        # Rows that hold the same payloads on purpose, because the datamatrix
        # is printed several times over. A payload found in one of them may
        # tick any cell of the group, which is how the second and third
        # printings are checked instead of being read as re-reads of the
        # first. None for an ordinary row, whose payloads are unique.
        self.group = group

    @property
    def is_dm(self):
        return self.kind == DATA_MATRIX

    def __repr__(self):
        what = " datamatrix" if self.is_dm else ""
        return f"<sheet row {self.number}{what}>"


class ValidationSheet:
    """The QR DATA columns of an xlsx, as an ordered list of rows."""

    def __init__(self, path, sheet=None):
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]

        raw = list(ws.iter_rows(values_only=True))
        if not raw:
            raise ValueError(f"{path}: sheet '{ws.title}' is empty")

        cols = []
        header = {}
        for idx, name in enumerate(raw[0]):
            if name is None:
                continue
            m = QR_COL_RE.match(str(name))
            if m:
                cols.append((int(m.group(1)), idx))
            else:
                header[str(name).strip().upper()] = idx
        if not cols:
            raise ValueError(f"{path}: no 'QR DATA<n>' columns in the header row")
        cols.sort()
        kind_at = header.get(TYPE_COL)
        print_at = header.get(PRINT_COL)
        source_at = header.get(SOURCE_COL)

        def cell(values, idx):
            if idx is None or idx >= len(values) or values[idx] is None:
                return ""
            return str(values[idx]).strip()

        self.path = path
        self.sheet_name = ws.title
        self.per_row = len(cols)
        self.rows = []
        self.by_text = {}      # payload -> [(row_index, col_index), ...]

        for number, values in enumerate(raw[1:], start=2):
            texts = []
            for _, idx in cols:
                v = values[idx] if idx < len(values) else None
                texts.append("" if v is None else str(v).strip())
            if not any(texts):
                continue                       # blank spacer row

            # A working copy says what each row is; an operator's own sheet
            # says nothing, and every row of it is QR.
            dm = cell(values, kind_at).upper().replace(" ", "") == "DATAMATRIX"
            source = cell(values, source_at)
            printing = cell(values, print_at)
            source = int(source) if source.isdigit() else None
            row = SheetRow(len(self.rows), number, texts,
                           kind=DATA_MATRIX if dm else QR_CODE,
                           printing=int(printing) if printing.isdigit() else None,
                           source=source,
                           # The printings of one datamatrix belong together,
                           # and what they have in common is the row of the
                           # operator's sheet they were expanded from. A copy
                           # written without that column falls back to the
                           # row itself, which groups nothing -- the right
                           # answer, since nothing says those rows are kin.
                           group=(f"dm{source if source is not None else number}"
                                  if dm else None))
            self.rows.append(row)
            for col, text in enumerate(texts):
                key = normalize(text)
                if not key:
                    continue
                self.by_text.setdefault(key, []).append((row.index, col))

        wb.close()
        self.dm_rows = sum(1 for r in self.rows if r.is_dm)
        # How many times an ordinary payload appears, which is what says the
        # sheet is one block of codes duplicated over and over. The
        # datamatrix rows are left out of it: they repeat on purpose, and
        # counting them would make every prepared sheet look duplicated.
        self.repeats = max((sum(1 for i, _c in v if not self.rows[i].is_dm)
                            for v in self.by_text.values()), default=0)
        print(f"[validate] {path} [{self.sheet_name}]: {len(self.rows)} rows x "
              f"{self.per_row} codes = {len(self.by_text)} unique payloads")
        if self.dm_rows:
            groups = len({r.group for r in self.rows if r.is_dm})
            print(f"[validate] {self.dm_rows} of those rows are datamatrix — "
                  f"{groups} value(s) per up, each printed "
                  f"{self.dm_rows // max(groups, 1)}x down the web")
        if self.repeats > 1:
            print(f"[validate] the sheet repeats: each payload appears up to "
                  f"{self.repeats} times, so a code is matched to the "
                  f"occurrence nearest the expected row")

    @property
    def has_dm(self):
        """Is there a datamatrix anywhere in this sheet? Decides whether the
        reader is asked to look for one at all."""
        return bool(self.dm_rows)

    def find(self, text, near=None):
        """(row_index, col_index) of this payload, or None.

        A sheet that repeats the same block of rows holds each payload many
        times over. `near` picks the occurrence closest to the row the
        sequence is expecting, so a repeated sheet walks straight through
        instead of snapping back to the first copy every cycle."""
        hits = self.by_text.get(normalize(text))
        if not hits:
            return None
        if near is None or len(hits) == 1:
            return hits[0]
        return min(hits, key=lambda h: abs(h[0] - near))

    def row(self, index):
        return self.rows[index] if 0 <= index < len(self.rows) else None


class Entry:
    """One label of a crossing: what was read, what belonged there, and where
    the code that was read actually comes from."""

    __slots__ = ("pos", "text", "expected", "status", "found")

    def __init__(self, pos, text, expected, status, found=None):
        self.pos = pos              # 0-based, top to bottom on screen
        self.text = text            # decoded payload, or None
        self.expected = expected    # payload the sheet wanted here
        self.status = status
        self.found = found          # (row_number, col_no) the payload is from

    @property
    def column(self):
        return f"QR DATA{self.pos + 1}"

    def describe(self):
        if self.status == SKIPPED:
            return f"{self.column}: not checked"
        if self.status == NODECODE:
            return f"{self.column}: no read, expected '{self.expected}'"
        if self.status == UNKNOWN:
            return f"{self.column}: '{self.text}' is not in the sheet at all"
        if self.status == SWAPPED:
            _, col = self.found
            return (f"{self.column}: holds the code for QR DATA{col} of this "
                    f"same row — labels are out of order")
        if self.status == WRONG_ROW:
            row, col = self.found
            return (f"{self.column}: holds row {row} QR DATA{col}, "
                    f"expected '{self.expected}'")
        return f"{self.column}: ok"


class BatchResult:
    """Verdict for one crossing."""

    def __init__(self, row, entries, anchored, complete, resynced, expected_row):
        self.row = row                  # spreadsheet row it was judged against
        self.entries = entries
        self.anchored = anchored
        self.complete = complete        # every label present and read
        self.resynced = resynced        # arrived out of sequence
        self.expected_row = expected_row

    @property
    def failures(self):
        return [e for e in self.entries if e.status not in (OK, SKIPPED)]

    @property
    def ok(self):
        return (self.anchored and self.complete and not self.resynced
                and not self.failures)

    def summary(self):
        """Short reason this crossing failed, for the machine-stop message and
        the results log. Says what actually went wrong rather than falling back
        to a catch-all word."""
        if not self.anchored:
            return "codes not in the sheet"
        if self.failures:
            first = self.failures[0]
            extra = f" (+{len(self.failures) - 1} more)" if len(self.failures) > 1 else ""
            return f"row {self.row} {first.column} {first.status}{extra}"
        if self.resynced:
            return f"row {self.row} out of sequence"
        if not self.complete:
            return f"row {self.row} incomplete ({len(self.entries)} labels)"
        return f"row {self.row}"

    @property
    def only_unread(self):
        """True when every fault is a label that did not read — nothing was
        wrong with the codes themselves."""
        return bool(self.failures) and all(e.status == NODECODE
                                           for e in self.failures)

    @property
    def has_wrong_code(self):
        """True if a checked position holds a real payload that belongs
        somewhere else — the one class of fault that should never be
        tolerated, because it means the wrong product is going out the door."""
        return any(e.status in (SWAPPED, WRONG_ROW, UNKNOWN)
                  for e in self.failures)

    @property
    def is_gap(self):
        """True for a failure that traces back to the *vision* side missing
        something — a label that never read, a row the camera never even
        detected (which surfaces as a clean resync with no bad codes in it),
        or a column that left the zone short a label — rather than a real
        wrong or misplaced code. Covers strictly more than `only_unread`: a
        resync where every code that *did* decode is correct for its own row
        is a pure detection gap too, not a mismatch, even though its
        `failures` list can be empty."""
        return not self.ok and not self.has_wrong_code


class SequenceValidator:
    """Check each crossing against the sheet, then step to the next row.

    The cursor holds the row expected next. It stays unset until a decoded
    payload is recognised; that code's own row anchors the sequence, and each
    completed crossing steps one row on. A crossing whose codes belong to some
    other row is reported as out-of-sequence and — unless resync is off — the
    cursor jumps there, so one skipped row doesn't cascade into every later
    row failing too.
    """

    def __init__(self, sheet, per_row=None, order=TOP_DOWN, resync=True,
                 check=None):
        self.sheet = sheet
        self.per_row = per_row or sheet.per_row
        self.order = order
        self.resync = resync
        # Which QR DATA positions are being checked, 0-based. None means all
        # of them; anything left out is neither read nor held against the row.
        self.check = None if check is None else set(check)
        self.cursor = None
        # Where a previous run got to, used only as the lookup hint until the
        # sequence anchors itself. A repeating sheet holds each payload many
        # times over, so without this the first code the camera sees resolves
        # to its copy in the very first block and the run starts again from
        # the top. Unlike setting the cursor outright, it does not make the
        # first crossing look out of sequence.
        self.resume_near = None

        self.batches = 0
        self.batches_ok = 0
        self.batches_bad = 0
        self.labels_ok = 0
        self.labels_bad = 0
        self.last = None
        self.exhausted = False

    @property
    def _near(self):
        """Which row to resolve a repeated payload against: where the sequence
        is now, or where the last run left off before it has anchored."""
        return self.cursor if self.cursor is not None else self.resume_near

    # ── live, per label: pure lookup, so decode order cannot mislead it ────
    def identify(self, text):
        """Row index this payload belongs to, or None — the decoder's notion
        of a column's identity. Two labels resolving to the same index are the
        same physical row, whatever the position tracker thinks."""
        hit = self.sheet.find(text, near=self._near)
        return None if hit is None else hit[0]

    def reanchor(self):
        """Forget where in the sheet we are, so the next crossing re-anchors.

        Called when the machine restarts: the web coasts and is nudged while
        it is stopped, so the row that comes back past the camera is not
        necessarily the one the sequence was expecting, and reporting that as
        a fault would just stop the machine again."""
        if self.cursor is not None:
            print("[validate] machine restarted — re-anchoring on the next row")
        self.cursor = None

    def is_checked(self, pos):
        return self.check is None or pos in self.check

    def describe_checks(self):
        if self.check is None:
            return f"all {self.per_row} positions"
        on = ", ".join(f"QR DATA{i + 1}" for i in sorted(self.check))
        off = ", ".join(f"QR DATA{i + 1}" for i in range(self.per_row)
                        if i not in self.check)
        return on + (f" (ignoring {off})" if off else "")

    def peek(self, text):
        """Where this payload lives in the sheet: (row_number, col_no) or None."""
        hit = self.sheet.find(text, near=self._near)
        if hit is None:
            return None
        row_idx, col = hit
        return self.sheet.rows[row_idx].number, col + 1

    # ── authoritative, per crossing ───────────────────────────────────────
    def check_batch(self, texts):
        """Judge one crossing. `texts` holds the payloads in on-screen order,
        top to bottom, with None where a label never read. Returns a
        BatchResult, or None if the crossing produced no reads at all."""
        if not any(texts):
            return self._unread_batch(texts)
        if self.order == BOTTOM_UP:
            texts = list(reversed(texts))

        hits = [self.sheet.find(t, near=self._near) if t else None
                for t in texts]
        seen = [row_idx for row_idx, _ in (h for h in hits if h)]

        if not seen:
            result = BatchResult(None, [Entry(i, t, None, UNKNOWN)
                                        for i, t in enumerate(texts)],
                                 False, False, False, None)
            self.last = result
            self.batches += 1
            return result

        observed = Counter(seen).most_common(1)[0][0]
        expected_row = self._anchor(observed)
        resynced = False
        if observed != expected_row:
            got = self.sheet.rows[observed]
            want = self.sheet.rows[expected_row]
            print(f"[validate] out of sequence: expected sheet row "
                  f"{want.number}, these labels are row {got.number}")
            if self.resync:
                print(f"[validate] resyncing to row {got.number}")
                expected_row = observed
            resynced = True
        self.cursor = expected_row

        row = self.sheet.rows[expected_row]
        entries = []
        for pos, text in enumerate(texts):
            expected = row.texts[pos] if pos < len(row.texts) else None
            if not self.is_checked(pos):
                entries.append(Entry(pos, text, expected, SKIPPED))
                continue
            if not text:
                entries.append(Entry(pos, None, expected, NODECODE))
                continue
            hit = hits[pos]
            if hit is None:
                entries.append(Entry(pos, text, expected, UNKNOWN))
                continue
            hit_row, hit_col = hit
            found = (self.sheet.rows[hit_row].number, hit_col + 1)
            if hit_row == expected_row and hit_col == pos:
                entries.append(Entry(pos, text, expected, OK, found))
            elif hit_row == expected_row:
                entries.append(Entry(pos, text, expected, SWAPPED, found))
            else:
                entries.append(Entry(pos, text, expected, WRONG_ROW, found))

        complete = len(texts) == self.per_row and all(
            text for pos, text in enumerate(texts) if self.is_checked(pos))
        result = BatchResult(row.number, entries, True, complete, resynced,
                             self.sheet.rows[expected_row].number)
        self.last = result
        self.batches += 1
        oks = sum(1 for e in entries if e.status == OK)
        self.labels_ok += oks
        self.labels_bad += sum(1 for e in entries
                               if e.status not in (OK, SKIPPED))
        if result.ok:
            self.batches_ok += 1
        else:
            self.batches_bad += 1
        self._advance()
        return result

    def _unread_batch(self, texts):
        """A column went by and not one of its labels read. That is a fault in
        its own right, but only once the sequence knows where it is — before
        that there is no row to hold it against."""
        if not texts or self.cursor is None:
            return None
        row = self.sheet.row(self.cursor)
        if row is None:
            return None
        entries = [Entry(i, None, row.texts[i] if i < len(row.texts) else None,
                         NODECODE if self.is_checked(i) else SKIPPED)
                   for i in range(len(texts))]
        if not any(e.status == NODECODE for e in entries):
            return None                      # nothing that was being checked
        result = BatchResult(row.number, entries, True, False, False, row.number)
        self.last = result
        self.batches += 1
        self.batches_bad += 1
        self.labels_bad += sum(1 for e in entries if e.status == NODECODE)
        self._advance()
        return result

    def _anchor(self, observed):
        if self.cursor is None:
            row = self.sheet.rows[observed]
            print(f"[validate] anchored on sheet row {row.number} "
                  f"('{row.texts[0]}') — sequence continues from there")
            self.cursor = observed
        return self.cursor

    def _advance(self):
        self.cursor += 1
        if self.cursor >= len(self.sheet.rows) and not self.exhausted:
            self.exhausted = True
            print("[validate] reached the end of the sheet — no rows left")

    def report(self, result):
        """Console summary of one crossing."""
        if result is None:
            return
        if not result.anchored:
            reads = [e.text for e in result.entries if e.text]
            print(f"[validate] ?? none of these codes are in the sheet: {reads}")
            return

        n = sum(1 for e in result.entries if e.status != SKIPPED)
        if result.ok:
            off = len(result.entries) - n
            note = f" ({off} not checked)" if off else ""
            print(f"[validate] PASS row {result.row}: {n}/{n} codes correct{note}")
            return

        why = []
        if result.failures:
            why.append(f"{len(result.failures)}/{n} wrong")
        # Labels seen is about the physical column — every position has to be
        # there for the ones being checked to line up, whether or not each is
        # itself being checked.
        if len(result.entries) != self.per_row:
            why.append(f"only {len(result.entries)} of {self.per_row} labels seen")
        if result.resynced:
            why.append("arrived out of sequence")
        print(f"[validate] FAIL row {result.row}: " + ", ".join(why))
        for entry in result.failures:
            print(f"           {entry.describe()}")

    # ── overlay ───────────────────────────────────────────────────────────
    def draw(self, frame, x=20, y=None):
        h = frame.shape[0]
        y = h - 150 if y is None else y
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.cursor is None:
            head, color = "VALIDATE: waiting for a known code", (0, 255, 255)
        else:
            row = self.sheet.row(self.cursor)
            nxt = f"row {row.number}" if row else "end of sheet"
            head = (f"VALIDATE: {self.batches_ok} pass / {self.batches_bad} fail"
                    f"  next {nxt}")
            color = (0, 0, 255) if self.batches_bad else (0, 255, 0)
        cv2.putText(frame, head, (x, y), font, 0.9, color, 2, cv2.LINE_AA)

        if self.last is not None:
            r = self.last
            if not r.anchored:
                line, lcolor = "last: no known codes", (0, 255, 255)
            elif r.ok:
                line, lcolor = f"last: row {r.row} PASS", (0, 255, 0)
            else:
                why = ", ".join(f"{e.column} {e.status}" for e in r.failures[:3])
                line, lcolor = f"last: row {r.row} FAIL - {why}", (0, 0, 255)
            cv2.putText(frame, line, (x, y + 36), font, 0.9, lcolor, 2, cv2.LINE_AA)
        return frame
