"""
ONNX -> TensorRT .engine, using the raw TensorRT Python API.
No ultralytics dependency here at all.

Usage:
    python build_engine.py --onnx best.onnx --engine best.engine --fp16
    python build_engine.py --onnx best.onnx --engine best.engine --int8 --calib-dir ./calib_images

Equivalent one-liner with trtexec, if you prefer the CLI tool that ships
with your TensorRT install:
    trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
"""
import argparse
import os

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4, min_shape=None,
                  opt_shape=None, max_shape=None, input_name="images"):
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("Warning: platform reports no fast fp16 support, building anyway")
        config.set_flag(trt.BuilderFlag.FP16)

    # Optional dynamic-shape profile (only needed if you exported with --dynamic)
    if min_shape and opt_shape and max_shape:
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    print("Building engine... this can take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved engine to {engine_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace", type=int, default=4, help="GB")
    ap.add_argument("--input-name", default="images")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()

    min_shape = opt_shape = max_shape = None
    if args.dynamic:
        min_shape = (1, 3, args.imgsz, args.imgsz)
        opt_shape = (1, 3, args.imgsz, args.imgsz)
        max_shape = (args.max_batch, 3, args.imgsz, args.imgsz)

    build_engine(
        args.onnx, args.engine, fp16=args.fp16, workspace_gb=args.workspace,
        min_shape=min_shape, opt_shape=opt_shape, max_shape=max_shape,
        input_name=args.input_name,
    )


if __name__ == "__main__":
    main()
