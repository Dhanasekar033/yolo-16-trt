# Label Inspector

Rolling-window QR validation for the label winder. A global-shutter camera
watches labels coming off the machine; every QR in frame is decoded and checked
against a spreadsheet, and the relay is cut the moment a row will not validate.

Self-contained — nothing outside this folder is needed.

## Run it

    python3 run_window.py

Nothing is decoded until you press **START** (or `s` in the window). START does
not energise the relay straight away: for `--start-delay` seconds the camera
reads the labels standing in front of it while the coil is still, and only then
does the relay go on. That read-in is what validates the coil at the position it
is actually in.

    q   quit          s   start          x   stop

## The re-inspection cycle

Nothing is tolerated. A row that reaches the end of the window still missing a
code is not written off — the line stops and the row is held open:

    row comes up short   ->  STOPPED, row held at the head of the window
    wind the coil back   ->  the same labels pass the camera again
    press START          ->  read-in re-reads them, then adjudicates:
                               filled in    -> cleared, the run carries on
                               still short  -> LABEL HAS ISSUE, recorded as a
                                               defect, window moves past it

## Common flags

    --window-size 4        how many sheet rows are open at once. Also the
                           detection latency: a short row is only noticed once
                           the web has run a whole window past it, and that is
                           how far back you have to wind. Smaller = sooner.
    --start-delay 2.0      seconds of reading after START before the relay
                           goes on. 0 = energise immediately.
    --xlsx validation.xlsx which sheet holds the expected QR DATA1..N codes.
    --check D2,D3          validate only those positions. Default: all of them.
    --xlsx-every 25        rewrite the annotated .xlsx every 25 rows, instead
                           of at exit only.
    --no-relay             vision only, no machine.
    --no-display           headless.
    --dump-crops bad/      save the crops that failed to decode.
    --debug                print every payload as it reads.

`python3 run_window.py --help` lists all of them.

## What it writes

Everything lands under `result/<sheet name>/`:

    progress.csv                 the journal. One line per row, appended and
                                 flushed as it happens. This is the durable
                                 record and what makes a run resumable.
    checked_<sheet>.xlsx         the deliverable: the source sheet with READ
                                 D1..N, ROW STATUS and CHECKED AT appended.
                                 Written at exit, or every --xlsx-every rows.
    labels/                      one JPEG per decoded label.
    runs/run_<stamp>.csv         per-row verdicts.

The source `--xlsx` is never modified.

Both files are read back on startup, so a run picks up where the last one left
off. Exit cleanly (`q`, or Ctrl-C) so the .xlsx gets written — killing the
process hard skips it, though nothing is lost because the journal has it all
and the next clean exit rebuilds the sheet from it.

## What it does not do

Codes are matched to the sheet one at a time, and the column a payload ticks
off comes from the sheet lookup, never from where the label sat. **Four correct
codes in the wrong four positions will pass.** If label position matters, this
is not the tool.

## Requires

Python 3, plus:

    opencv-python   built with GStreamer support
    tensorrt        matching the GPU the .engine was built on
    cuda-python
    numpy  openpyxl  pyserial  zxing-cpp

`best.engine` is built for a specific GPU. On different hardware, rebuild it
from the ONNX model rather than copying this one.
