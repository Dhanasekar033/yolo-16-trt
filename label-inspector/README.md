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

Nothing is tolerated. A row the coil has moved past while it is still missing
a code is not written off — the line stops and the row is held open:

    row comes up short   ->  STOPPED, row held at the head of the window
    wind the coil back   ->  the same labels pass the camera again
    press START          ->  read-in re-reads them, then adjudicates:
                               filled in    -> cleared, the run carries on
                               still short  -> LABEL HAS ISSUE, recorded as a
                                               defect, window moves past it

The model detects three classes — `label`, `qr_code` and `logo`. The code is
cropped with a quiet zone (`--qr-margin`) and handed to the reader; the logo
is never decoded, it only has to be present. A label found without its code or
its logo stops the line on that frame, marked in red on the picture, and the
line restarts once it has been wound out of shot.

A label that will not read at all is the quietest failure there is: it is not
a match, not a repeat and not an unexpected code, so it moves no part of the
machine. If labels keep coming past and not one of them gives up a code for
`--no-read-secs` (1.5s), the line stops — the labels may carry no QR, or the
lens or the light may have gone. It starts itself again the moment one label
reads. A single blank row among readable ones is caught by the window
instead, the same way any short row is.

If **no** code read is anywhere in the sheet, the roll is not the one the
sheet describes — a new roll went on and the sheet was not changed with it.
The line stops before the relay is energised at all, the console reads WRONG
SHEET, and LOAD SHEET stays live so the right sheet can be loaded; loading it
starts a fresh record and the run carries on normally.

## Common flags

    --hold-after 2         how many rows the coil may move past a row that
                           did not read everything before the line stops for
                           it. This is the detection latency, and how far back
                           you have to wind. 1 stops at the very next row;
                           raise it if labels from several rows are in view at
                           once, so a row's last code has time to arrive.
    --window-size 8        how many sheet rows a code can still be recognised
                           from. Bigger tolerates more out-of-order arrival;
                           it no longer decides how long a short row runs on.
    --no-part-check        do not stop when a label is missing its QR or its
                           logo. On by default: the model finds all three, so
                           a missing part is caught on the label itself.
    --no-read-secs 1.5     stop if labels are in frame this long and not one
                           of them reads. 0 turns the watchdog off.
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

The paperwork stays in the project, under `result/<sheet name>/`:

    progress.csv                 the journal. One line per row, appended and
                                 flushed as it happens. This is the durable
                                 record and what makes a run resumable.
    checked_<sheet>.xlsx         the deliverable: the source sheet with READ
                                 D1..N, ROW STATUS and CHECKED AT appended.
                                 Written at exit, or every --xlsx-every rows.
    runs/run_<stamp>.csv         per-row verdicts.

The label crops go somewhere of their own, because they are bulk image data
and they fill a disk: `--label-dir`, `labels/` by default, one folder per
sheet and nothing in it but pictures. Each crop is the detection box exactly
as the model drew it — nothing is padded unless `--label-pad` asks for it:

    <label-dir>/<sheet name>/    one JPEG per decoded label.

The LABEL FOLDER button on the console repoints that, and the choice is
remembered between runs (in `~/.config/label-inspector/settings.json`, along
with the sheets that have been loaded — OPEN RECENT SHEET lists them, and the
last one is reopened at startup when no `--xlsx` is given).

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
