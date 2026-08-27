# YOLO26 → TensorRT, ultralytics-free inference

Same idea as [Linaom1214/TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series),
updated for YOLO26 and the modern TensorRT 10.x Python API (`execute_async_v3`,
`set_tensor_address`).

`ultralytics` is only imported in **`export_onnx.py`** (step 1). Everything
after that — engine build and inference — uses just `tensorrt`, `cuda-python`,
`numpy`, `opencv-python`.

## Why this is simpler than YOLOv8/v11 versions of this pattern

YOLO26 has a NMS-free, end-to-end head. If you export with `end2end=True`,
the ONNX/engine's single output is already `(batch, max_det, 6)` =
`[x1, y1, x2, y2, conf, class_id]` in letterboxed-input pixel space, with
NMS baked into the graph. So the inference code here never has to
reimplement NMS/decoding — just a confidence filter and rescale back to
the original image.

## 1. Export .pt → ONNX (needs ultralytics, one-time, can be on any machine)

```bash
pip install ultralytics
python export_onnx.py --weights best.pt --imgsz 640 --half
```

## 2. Build the TensorRT engine

Must be run **on the target deploy machine** — engines are architecture/
TensorRT-version specific. A desktop RTX engine won't run on a Jetson and
vice versa, and same-family Jetson SKUs aren't guaranteed portable either.

```bash
pip install tensorrt   # or use the TensorRT that ships with JetPack
python build_engine.py --onnx best.onnx --engine best.engine --fp16
```

or with the `trtexec` CLI that ships with your TensorRT install:

```bash
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

## 3. Run inference — no ultralytics import anywhere below this line

```bash
pip install "numpy<2" cuda-python opencv-python
```

**Static image / video file / plain webcam:**
```bash
python3 detect.py --engine best.engine --source image.jpg --classes classes.txt --save out.jpg
python3 detect.py --engine best.engine --source video.mp4 --save out.mp4
python3 detect.py --engine best.engine --source 0 --show
```

**Global Shutter Camera via GStreamer (auto-detects the device, fixed
rotation, live FPS overlay):**
```bash
python3 cam_infer.py --engine best.engine
python3 cam_infer.py --engine best.engine --classes classes.txt --conf-thres 0.35
python3 cam_infer.py --engine best.engine --save out.mp4
python3 cam_infer.py --engine best.engine --no-display   # headless
```

`classes.txt` = one class name per line, in the same order as your training
`data.yaml` `names:` list.

## Files

| file | role |
|---|---|
| `export_onnx.py` | .pt → .onnx (only file that imports ultralytics) |
| `build_engine.py` | .onnx → .engine, raw TensorRT builder API |
| `trt_engine.py` | loads engine, allocates GPU buffers, runs inference (`YOLO26TRT` class) |
| `utils.py` | letterbox preprocessing + box rescaling/drawing |
| `detect.py` | CLI for image/video/plain-webcam sources |
| `cam_infer.py` | CLI for the Global Shutter Camera GStreamer pipeline, with live inference overlay |

## Environment notes / gotchas

- **Engines are not portable.** Rebuild on the exact GPU + TensorRT version
  you'll deploy on (desktop GPU, Jetson Orin, Jetson Thor, etc. all need
  their own build).
- **NumPy 2.x breaks OpenCV wheels compiled against NumPy 1.x**
  (`AttributeError: _ARRAY_API not found`). Pin with `pip install "numpy<2"`,
  or reinstall opencv-python if you need to stay on NumPy 2.x.
- **Why cuda-python instead of pycuda:** pycuda has to be *compiled* on
  install and needs the full CUDA toolkit dev headers (`cuda.h`) on your
  include path, which many setups (driver-only installs, some Jetson
  images) don't have. `cuda-python` ships prebuilt wheels — `pip install
  cuda-python` — with nothing to compile.
- **cuda-python import path changed in v12.8+**: `cudart` moved from
  `cuda.cudart` to `cuda.bindings.runtime`. `trt_engine.py` already handles
  both via a try/except import, so this shouldn't bite you, but if you see
  `ImportError: cannot import name 'cudart' from 'cuda'` on some other
  script, that's why.
- If you didn't export with `end2end=True`, the output tensor is raw
  `(batch, num_boxes, 4+nc)` predictions and you'd need to add your own
  NMS (the classic path the old TensorRT-For-YOLO-Series repo used) —
  say the word if that's your situation and I'll add that variant.
- Dynamic batch: export ONNX with `--dynamic`, then pass matching
  `--dynamic --imgsz 640 --max-batch N` to `build_engine.py`.
- This code was written and reviewed without a GPU/TensorRT available in
  the sandbox that produced it — it's been shaped by your actual run
  output so far, but if something else doesn't line up, paste the error
  and I'll fix it.