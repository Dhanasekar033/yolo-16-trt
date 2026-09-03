# Backup: the OpenCV/GTK console

The state of the app just before the PyQt front end was added.

    run.py.opencv-ui     ->  run.py
    ui.py.opencv-ui      ->  utils/ui.py

To go back to it exactly:

    cp backup/run.py.opencv-ui run.py
    cp backup/ui.py.opencv-ui  utils/ui.py

Nothing else changed, so no other file needs restoring.

You should not normally need this: the OpenCV console is still in the current
build and still works. `run.py --ui opencv` selects it, and `--ui qt` (the
default) selects the Qt one.
