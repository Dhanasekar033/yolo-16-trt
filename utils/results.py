"""A CSV record of every crossing the line produced.

One row per verdict, written and flushed as it happens, so the file is
complete even if the run is killed. It lands next to the label crops:

    result/<xlsx name>/runs/run_<timestamp>.csv
"""

import csv
import os
import time

HEADER = ["time", "sheet_row", "verdict", "reason", "labels", "read",
          "d1", "d2", "d3", "d4"]


class ResultLog:
    def __init__(self, root="result", name="run", subdir="runs", columns=4):
        folder = os.path.join(root, name, subdir)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = os.path.join(folder, f"run_{stamp}.csv")
        self.columns = columns
        self.rows = 0

        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(HEADER[:6] + [f"d{i + 1}" for i in range(columns)])
        self._fh.flush()
        print(f"[results] logging verdicts to {self.path}")

    def write(self, result):
        if result is None:
            return
        cells = []
        for i in range(self.columns):
            entry = next((e for e in result.entries if e.pos == i), None)
            if entry is None:
                cells.append("")
            elif entry.status == "OK":
                cells.append("ok")
            elif entry.status == "SKIPPED":
                cells.append("off")
            else:
                cells.append(entry.status)

        self._writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            result.row if result.row is not None else "",
            "PASS" if result.ok else "FAIL",
            "" if result.ok else result.summary(),
            len(result.entries),
            sum(1 for e in result.entries if e.text),
        ] + cells)
        self._fh.flush()
        self.rows += 1

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
