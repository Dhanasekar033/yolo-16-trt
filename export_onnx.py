"""
One-time export step: PyTorch (.pt) -> ONNX.

This is the ONLY step that touches the `ultralytics` package. Everything
after this (engine build + inference) uses plain TensorRT/pycuda/OpenCV,
no ultralytics import at all.

Key point for YOLO26: export with end2end=True. YOLO26 has a NMS-free
head, so an end2end export bakes NMS into the graph and the ONNX/TRT
output is already (batch, max_det, 6) = [x1, y1, x2, y2, conf, cls] in
input-image (letterboxed) pixel coordinates. That means the standalone
inference code below does NOT need to reimplement NMS.

Usage:
    python export_onnx.py --weights best.pt --imgsz 640 --half
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to trained yolo26 .pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="export in fp16")
    ap.add_argument("--dynamic", action="store_true", help="dynamic batch axis")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--opset", type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO  # only needed right here, for export

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        simplify=True,
        end2end=True,        # bakes NMS in -> output (batch, max_det, 6)
        max_det=args.max_det,
        opset=args.opset,
    )
    print(f"Exported ONNX model to: {onnx_path}")
    print("Next: build the TensorRT engine with build_engine.py")


if __name__ == "__main__":
    main()
