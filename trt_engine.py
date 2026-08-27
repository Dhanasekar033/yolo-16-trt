"""
Pure TensorRT + cuda-python inference wrapper for a YOLO26 end2end engine.
No ultralytics import anywhere in this file.

Uses NVIDIA's official `cuda-python` package (prebuilt wheels, nothing to
compile) instead of pycuda, to avoid needing the CUDA toolkit dev headers
installed just to build a driver-binding wheel.

    pip install cuda-python

Assumes the engine was exported with end2end=True, so the single output
tensor has shape (batch, max_det, 6) = [x1, y1, x2, y2, conf, class_id]
already in input-tensor pixel coordinates, NMS already applied.
"""
import numpy as np
import tensorrt as trt

# cuda-python restructured its layout in v12.8+: cudart moved from
# `cuda.cudart` to `cuda.bindings.runtime`. Try the new path first, fall
# back to the old one so this works on either version.
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def _check(err):
    if isinstance(err, tuple):
        err = err[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


class YOLO26TRT:
    def __init__(self, engine_path, input_size=(640, 640)):
        self.input_size = input_size  # (h, w)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        _check(err)

        self.input_name = None
        self.output_name = None
        self.host_buffers = {}
        self.device_buffers = {}
        self.nbytes = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

            # resolve dynamic (-1) batch dim to 1 for buffer allocation
            shape = tuple(1 if d == -1 else d for d in shape)

            host_mem = np.empty(shape, dtype=dtype)
            err, device_ptr = cudart.cudaMalloc(host_mem.nbytes)
            _check(err)

            self.host_buffers[name] = host_mem
            self.device_buffers[name] = device_ptr
            self.nbytes[name] = host_mem.nbytes
            self.context.set_tensor_address(name, int(device_ptr))

            if is_input:
                self.input_name = name
                self.input_dtype = dtype
                if -1 in self.engine.get_tensor_shape(name):
                    self.context.set_input_shape(name, shape)
            else:
                self.output_name = name

    def infer(self, input_array):
        """input_array: contiguous np.float32/float16 array shaped (1,3,H,W)"""
        input_array = np.ascontiguousarray(input_array, dtype=self.input_dtype)
        self.host_buffers[self.input_name] = input_array

        _check(cudart.cudaMemcpyAsync(
            self.device_buffers[self.input_name],
            input_array.ctypes.data,
            input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            self.stream,
        ))

        self.context.execute_async_v3(stream_handle=self.stream)

        out_host = self.host_buffers[self.output_name]
        _check(cudart.cudaMemcpyAsync(
            out_host.ctypes.data,
            self.device_buffers[self.output_name],
            self.nbytes[self.output_name],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            self.stream,
        ))
        _check(cudart.cudaStreamSynchronize(self.stream))
        return out_host  # shape (1, max_det, 6)

    def __del__(self):
        for ptr in getattr(self, "device_buffers", {}).values():
            try:
                cudart.cudaFree(ptr)
            except Exception:
                pass
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                cudart.cudaStreamDestroy(stream)
            except Exception:
                pass
