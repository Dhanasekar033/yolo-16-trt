"""A CSV of what was read, written beside the crops it describes.

One line per code accepted, in the order the camera read them, in the same
folder the label crops go to -- so a folder of pictures is never handed over
without the record of what each one was and where it belongs.

    <label-dir>/<xlsx name>/scan_001.csv
    <label-dir>/<xlsx name>/scan_002.csv
    ...

THE FILE IS SPLIT AT EVERY DATAMATRIX, and that is the whole point of this
module. On this reel a datamatrix is not another code to check -- it is the
mark between one job and the next, printed where the QR rows stop. A single
CSV covering a whole shift would run the jobs together, and separating them
afterwards means reading down thousands of lines looking for the row where the
kind changes. So a datamatrix closes the file it appears in, as its last line,
and the next QR opens a new one. Each file is then one job: the codes between
two marks, with the mark that ended it at the bottom.

Nothing rolls until a QR actually arrives. A datamatrix is printed several
times over and every printing is read, so closing on the mark and opening on
the next mark would leave a file per printing with nothing in it.

The file is flushed after every line. A panel PC next to a winder loses power
the way machines do, and a log that only reaches the disk when the app closes
cleanly is a log that is missing exactly the run somebody wants to look at.
"""

import csv
import os
import re
import time

QR = "qr"
DATAMATRIX = "datamatrix"

FIELDS = ("time", "kind", "value", "id", "sheet_row", "up", "column", "image")

NUMBERED = re.compile(r"^(.*)_(\d+)\.csv$", re.IGNORECASE)


class ScanLog:
    """The run's CSVs, and which one is open now."""

    def __init__(self, root="labels", name="run", prefix="scan", fields=FIELDS):
        self.dir = os.path.join(root, name)
        self.prefix = prefix
        # The columns this run writes. A build with two cameras adds one for
        # which of them read the code; a one-camera run has nothing to say
        # there, and a column of blanks in its record would be a question
        # nobody asked.
        self.fields = tuple(fields)
        self.total = 0           # lines written by this run, across all files
        self.count = 0           # lines in the file that is open
        self.files = 0           # files this run has opened
        self.last = None         # the last line written, as a dict
        self.path = None
        self._fh = None
        self._writer = None
        # Set by a datamatrix and cleared by the QR that opens the next file.
        self._ended = False
        # Where the numbering carries on from. A folder that already holds
        # scan_001..scan_004 belongs to an earlier run over the same sheet,
        # and starting again at 001 would either overwrite it or interleave
        # two runs in one file.
        self._next = self._survey() + 1

    def _survey(self):
        """The highest file number already in the folder, or 0."""
        top = 0
        try:
            names = os.listdir(self.dir)
        except OSError:
            return 0
        for name in names:
            hit = NUMBERED.match(name)
            if hit and hit.group(1) == self.prefix:
                top = max(top, int(hit.group(2)))
        return top

    def _open(self):
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, f"{self.prefix}_{self._next:03d}.csv")
        self._next += 1
        self.files += 1
        self.count = 0
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields)
        self._writer.writeheader()
        self._fh.flush()
        print(f"[scanlog] writing {self.path}")

    def log(self, kind, value, code_id="", sheet_row=None, up=None,
            column=None, image="", **extra):
        """Write one code. `kind` is QR or DATAMATRIX.

        Returns the line as a dict, which is what the console shows as the
        last thing scanned -- read from the log rather than from the crop
        saver so that what the screen says and what the record says can never
        disagree.
        """
        if kind == QR and self._ended:
            self.close()             # the datamatrix before it ended this job
        if self._fh is None:
            self._open()

        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S")
                       + f".{int(time.time() % 1 * 1000):03d}",
               "kind": kind,
               "value": value or "",
               "id": code_id or "",
               "sheet_row": "" if sheet_row is None else sheet_row,
               "up": "" if up is None else f"UP{up + 1}",
               "column": "" if column is None else f"QR DATA{column + 1}",
               "image": image or ""}
        # Only the columns this log declared: an extra a caller passes that
        # this run has no column for is dropped rather than breaking the file.
        row.update({k: v for k, v in extra.items() if k in self.fields})
        row = {k: row.get(k, "") for k in self.fields}
        self._writer.writerow(row)
        # Flushed line by line: see the note at the top of the file.
        self._fh.flush()
        self.count += 1
        self.total += 1
        row["file"] = os.path.basename(self.path)
        self.last = row
        if kind == DATAMATRIX:
            # The mark closes the job it ends. The file itself stays open
            # until the next QR, so a second printing of the same mark lands
            # in it rather than in a file of its own.
            self._ended = True
        return row

    def close(self):
        if self._fh is not None:
            self._fh.close()
            print(f"[scanlog] {self.count} line(s) in {self.path}")
        self._fh = self._writer = None
        self._ended = False
