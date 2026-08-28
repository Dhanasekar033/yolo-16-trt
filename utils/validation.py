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

OK        = "OK"
SWAPPED   = "SWAPPED"     # right row, but sitting in another column
WRONG_ROW = "WRONG-ROW"   # a real code, but from a different sheet row
UNKNOWN   = "UNKNOWN"     # decoded fine, but no such code in the sheet
NODECODE  = "NO-READ"     # a label was there, but it never read

TOP_DOWN  = "top-down"
BOTTOM_UP = "bottom-up"


def normalize(text):
    """Compare payloads case- and whitespace-insensitively."""
    return (text or "").strip().upper()


class SheetRow:
    """One spreadsheet row: the QR DATA1..N payloads expected side by side."""

    __slots__ = ("index", "number", "texts")

    def __init__(self, index, number, texts):
        self.index = index        # 0-based position in the sequence
        self.number = number      # 1-based spreadsheet row number
        self.texts = texts        # [QR DATA1, QR DATA2, ...]

    def __repr__(self):
        return f"<sheet row {self.number}>"


class ValidationSheet:
    """The QR DATA columns of an xlsx, as an ordered list of rows."""

    def __init__(self, path, sheet=None):
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = wb[sheet] if sheet else wb.worksheets[0]

        raw = list(ws.iter_rows(values_only=True))
        if not raw:
            raise ValueError(f"{path}: sheet '{ws.title}' is empty")

        cols = []
        for idx, name in enumerate(raw[0]):
            m = QR_COL_RE.match(str(name)) if name is not None else None
            if m:
                cols.append((int(m.group(1)), idx))
        if not cols:
            raise ValueError(f"{path}: no 'QR DATA<n>' columns in the header row")
        cols.sort()

        self.path = path
        self.sheet_name = ws.title
        self.per_row = len(cols)
        self.rows = []
        self.by_text = {}
        duplicates = []

        for number, values in enumerate(raw[1:], start=2):
            texts = []
            for _, idx in cols:
                v = values[idx] if idx < len(values) else None
                texts.append("" if v is None else str(v).strip())
            if not any(texts):
                continue                       # blank spacer row

            row = SheetRow(len(self.rows), number, texts)
            self.rows.append(row)
            for col, text in enumerate(texts):
                key = normalize(text)
                if not key:
                    continue
                if key in self.by_text:
                    duplicates.append(text)
                else:
                    self.by_text[key] = (row.index, col)

        wb.close()
        print(f"[validate] {path} [{self.sheet_name}]: {len(self.rows)} rows x "
              f"{self.per_row} codes = {len(self.by_text)} unique payloads")
        if duplicates:
            print(f"[validate] warning: {len(duplicates)} payloads appear more "
                  f"than once — placement checks on those will be unreliable")

    def find(self, text):
        """(row_index, col_index) of this payload in the sheet, or None."""
        return self.by_text.get(normalize(text))

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
        return [e for e in self.entries if e.status != OK]

    @property
    def ok(self):
        return (self.anchored and self.complete and not self.resynced
                and not self.failures)


class SequenceValidator:
    """Check each crossing against the sheet, then step to the next row.

    The cursor holds the row expected next. It stays unset until a decoded
    payload is recognised; that code's own row anchors the sequence, and each
    completed crossing steps one row on. A crossing whose codes belong to some
    other row is reported as out-of-sequence and — unless resync is off — the
    cursor jumps there, so one skipped row doesn't cascade into every later
    row failing too.
    """

    def __init__(self, sheet, per_row=None, order=TOP_DOWN, resync=True):
        self.sheet = sheet
        self.per_row = per_row or sheet.per_row
        self.order = order
        self.resync = resync
        self.cursor = None

        self.batches = 0
        self.batches_ok = 0
        self.batches_bad = 0
        self.labels_ok = 0
        self.labels_bad = 0
        self.last = None
        self.exhausted = False

    # ── live, per label: pure lookup, so decode order cannot mislead it ────
    def peek(self, text):
        """Where this payload lives in the sheet: (row_number, col_no) or None."""
        hit = self.sheet.find(text)
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
            return None
        if self.order == BOTTOM_UP:
            texts = list(reversed(texts))

        hits = [self.sheet.find(t) if t else None for t in texts]
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

        complete = len(texts) == self.per_row and all(texts)
        result = BatchResult(row.number, entries, True, complete, resynced,
                             self.sheet.rows[expected_row].number)
        self.last = result
        self.batches += 1
        oks = sum(1 for e in entries if e.status == OK)
        self.labels_ok += oks
        self.labels_bad += len(entries) - oks
        if result.ok:
            self.batches_ok += 1
        else:
            self.batches_bad += 1
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

        n = len(result.entries)
        if result.ok:
            print(f"[validate] PASS row {result.row}: {n}/{n} codes correct")
            return

        why = []
        if result.failures:
            why.append(f"{len(result.failures)}/{n} wrong")
        if n != self.per_row:
            why.append(f"only {n} of {self.per_row} labels seen")
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
