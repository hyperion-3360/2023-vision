import argparse
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image, ImageDraw


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Get console logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    Formatter = logging.Formatter
    formatter = Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname).1s] %(message)s",
        datefmt="%d/%m/%Y-%H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


# Default logger
logger = get_logger("TRTDemo")

MIN_SIZE = (416, 416)
MAX_SIZE = (1080, 1920)


class NamedBinding(NamedTuple):
    name: str
    dtype: np.dtype
    shape: Tuple[int, ...]
    data: np.ndarray
    device: cuda.DeviceAllocation


class Timer:
    PRECISION = {"s": 1, "ms": 1e3, "us": 1e6}

    def __init__(self, name: str, precision: str = "ms") -> None:
        assert precision in self.PRECISION
        self.name = name
        self.precision = precision
        self.reset()

    def start(self) -> None:
        self._start = time.time()

    def stop(self) -> None:
        self._end = time.time()

    def reset(self) -> None:
        self._start = 0
        self._end = 0

    def report(self) -> float:
        mult = self.PRECISION[self.precision]
        total = (self._end - self._start) * mult
        logger.info(f"{self.name} took {total:.2f}{self.precision}")
        return total


def timeit(name: str, precision: str = "ms"):
    """Function timer decorator."""
    timer = Timer(name, precision=precision)

    def timeit_decorator(func):
        @wraps(func)
        def timeit_fn(*args, **kwargs):
            timer.start()
            ret = func(*args, **kwargs)
            timer.stop()
            timer.report()
            return ret

        return timeit_fn

    return timeit_decorator


class YOLOXTensorRT:
    """
    YOLOX TensorRT inference engine.

    :param engine_path: serialized TensorRT engine path
    :param precision: model datatype (either "float16" or "float32")
    :param input_shape: input tensors shape (required with dynamic input dimensions)
    """

    def __init__(
        self,
        engine_path: str,
        precision: str = "float32",
        input_shape: Optional[Sequence[int]] = None,
    ) -> None:
        assert precision in ("float16", "float32")
        self.dtype = np.float32 if precision == "float32" else np.float16
        self.bindings: Dict[str, NamedBinding] = {}
        self.binding_addrs: Dict[str, int] = {}
        self.engine = self._load_engine(engine_path)
        self.context = self._init_context(input_shape=input_shape)
        self.stream = cuda.Stream()
        self.batch_size = 1

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load TensorRT engine."""
        # Initialize and register all the existing TensorRT plugins (e.g., BatchedNMS)
        trt_logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(trt_logger, "")

        with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        logger.info(f"Loaded {engine_path} for TensorRT inference")

        return engine

    def _init_context(
        self, input_shape: Optional[Sequence[int]] = None
    ) -> trt.IExecutionContext:
        """Initialize inference execution context."""

        # Context managers used in TensorRT samples are deprecated and have no effect.
        # Objects are automatically freed when the reference count reaches 0.
        context = self.engine.create_execution_context()

        for index in range(self.engine.num_bindings):
            # Binding properties
            name = self.engine.get_binding_name(index)
            shape = tuple(self.engine.get_binding_shape(index))
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            is_input = self.engine.binding_is_input(index)

            dynamic_axes = shape.count(-1)
            if dynamic_axes > 0 and is_input:
                if input_shape is None or len(input_shape) != dynamic_axes:
                    raise ValueError("Cannot allocate buffers for dynamic input shape")

                input_shape = list(input_shape).copy()
                shape = tuple(s if s != -1 else input_shape.pop(0) for s in shape)
                # Set dynamic dimensions for input tensor
                context.set_binding_shape(index, shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(shape, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Register the named binding
            self.bindings[name] = NamedBinding(name, dtype, shape, host_mem, device_mem)

        self.binding_addrs = {n: int(d.device) for n, d in self.bindings.items()}

        return context

    @timeit("Inference")
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Execute inference on input `image`.
        Expected input is a row-major order ("C" order) numpy array with NCHW format.

        Returns a tuple of size `batch_size` containing prediction arrays with shape
        [num_detections, 6]. Bounding box coordinates are normalized between [0, 1].
        Each detection consists in 6 values: (class_pred, conf_score, x1, y1, x2, y2).

        :param image: input image
        :returns: tuple of prediction arrays
        """
        # Set correct input dtype
        if image.dtype != self.dtype:
            image = image.astype(self.dtype)

        # Copy image to pagelocked memory
        inp = self.bindings["input"]
        np.copyto(inp.data, image)
        # Transfer input data to the GPU
        cuda.memcpy_htod_async(inp.device, inp.data, self.stream)
        # Run inference
        self.context.execute_async_v2(
            list(self.binding_addrs.values()), stream_handle=self.stream.handle
        )
        # Transfer predictions back from the GPU
        [
            cuda.memcpy_dtoh_async(
                self.bindings[out].data, self.bindings[out].device, self.stream
            )
            for out in ("num_boxes", "boxes")
        ]
        # Synchronize the stream
        self.stream.synchronize()

        # Reshape outputs
        rc = []

        ndet = self.bindings["num_boxes"].data.reshape(self.batch_size)

        if ndet.item() is not 0:
            preds = self.bindings["boxes"].data[:,:ndet.item(),:]
            cleaned = np.reshape(preds, preds.shape[1:])

            boxes = cleaned[:,2:6]
            scores = cleaned[:,1]
            classes= cleaned[:,0]

            for i in np.concatenate((classes[:,None], scores[:,None], boxes), axis=1):
                rc.append(tuple(i))

        return rc

    def warmup(self, n: int = 1) -> None:
        """Warmup model by running inference `n` times."""
        logger.info("-" * 32)
        logger.info(f"TensorRT model warmup with n={n}")
        inp = self.bindings["input"]
        for _ in range(n):
            image = np.zeros_like(inp.data)
            self(image)
        logger.info("-" * 32)

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.

    :param verbose: enable a higher verbosity level will be set on the TensorRT logger
    :param workspace: max memory workspace to allow (in Gb)
    :param precision: model datatype (either "float16" or "float32")
    """

    def __init__(
        self,
        verbose: bool = False,
        workspace: int = 4,
        precision: str = "float32",
    ) -> None:
        assert precision in ("float16", "float32")
        self.logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.logger.min_severity = trt.Logger.Severity.VERBOSE
        self.max_workspace_size = workspace * 1 << 30
        self.precision = precision
        self.batch_size = 1

        # Initialize and register all the existing TensorRT plugins (e.g., BatchedNMS)
        trt.init_libnvinfer_plugins(self.logger, namespace="")

    def create_engine(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        opt_size: Tuple[int, int],
        min_size: Tuple[int, int] = MIN_SIZE,
        max_size: Tuple[int, int] = MAX_SIZE,
    ) -> None:
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition,
        which is saved to disk.

        :param onnx_path: ONNX model path
        :param engine_path: serialized engine save path
        :param opt_size: optimum/expected input image size
        :param min_size: minimum input image size
        :param max_size: maximum input image size
        """
        logger.info(f"Building {self.precision} TRTEngine in '{engine_path}'")
        with trt.Builder(self.logger) as builder:
            # Set expected input dimensions for dynamic shapes
            config = builder.create_builder_config()
            profile = builder.create_optimization_profile()

            # Set the minimum/optimum/maximum values for a shape input tensor
            dims = (self.batch_size, 3)
            profile.set_shape("input", dims + min_size, dims + opt_size, dims + max_size)
            config.add_optimization_profile(profile)

            if self.precision == "float16":
                if not builder.platform_has_fast_fp16:
                    logger.warning(
                        "FP16 is not natively supported on this platform/device"
                    )
                else:
                    config.set_flag(trt.BuilderFlag.FP16)
                    config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            with builder.create_network(flag) as network, trt.OnnxParser(
                network, self.logger
            ) as parser:
                config.max_workspace_size = self.max_workspace_size
                # builder.max_batch_size = self.batch_size
                # Parse model from buffer
                if not parser.parse_from_file(str(onnx_path)):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError(f"Failed to load ONNX file '{onnx_path}'")

                inputs = [network.get_input(i) for i in range(network.num_inputs)]
                outputs = [network.get_output(i) for i in range(network.num_outputs)]

                # Network description
                logger.info("-" * 64)
                logger.info(f"Network Description: {network.name}")
                for i, input in enumerate(inputs):
                    logger.info(
                        f"Input_{i}(name='{input.name}', shape={input.shape}, dtype={input.dtype})"
                    )
                for i, output in enumerate(outputs):
                    logger.info(
                        f"Output_{i}(name='{output.name}', shape={output.shape}, dtype={output.dtype})"
                    )
                logger.info("-" * 64)

                logger.info("Building an engine from file, this may take a while...")
                with builder.build_serialized_network(network, config) as engine:
                    engine_path = Path(engine_path)
                    engine_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(engine_path, "wb") as f:
                        f.write(engine)
                logger.info(f"Serialized engine successfully saved at '{engine_path}'")

def export_tensorrt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    opt_size: Tuple[int, int] = (608, 608),
    precision: str = "float32",
    verbose: bool = False,
    workspace: int = 4,
) -> None:
    """
    Export ONNX model and TensorRT serialized engine for TensorRT inference.

    :param onnx_path: ONNX model path
    :param engine_path: serialized engine save path
    :param opt_size: optimum/expected input image size
    :param precision: model datatype (either "float16" or "float32")
    :param verbose: enable a higher verbosity level will be set on the TensorRT logger
    :param workspace: max memory workspace to allow (in Gb)
    """
    # Set the default TensorRT engine path
    engine_path = engine_path or Path(onnx_path).with_suffix(".trt")

    # Build and export the TensorRT engine
    engine_builder = EngineBuilder(
        verbose=verbose, workspace=workspace, precision=precision
    )
    engine_builder.create_engine(onnx_path, engine_path, opt_size)


def convert_to_nchw(
    image: Union[Image.Image, np.ndarray], dtype: Union[np.dtype, str] = "float32"
) -> np.ndarray:
    """Convert `image` from HWC format to NCHW."""
    # HWC to CHW format
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.array(image, dtype=dtype, order="C")
    return image


def draw_bboxes(
    image: Image.Image, boxes: np.ndarray, categories: List[str], color: str = "blue"
) -> Image.Image:
    """Draw the bounding boxes on the original input image and return it."""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        category, score, x1, y1, x2, y2 = box
        left = max(0, np.floor(x1 * image.width + 0.5).astype(int))
        top = max(0, np.floor(y1 * image.height + 0.5).astype(int))
        right = min(image.width, np.floor(x2 * image.width + 0.5).astype(int))
        bottom = min(image.height, np.floor(y2 * image.height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=color)
        draw.text(
            (left, top - 12), f"{categories[int(category)]} {score:.2f}", fill=color
        )

    return image


def draw_outputs(
    image: Image.Image, output_file: str, boxes , categories: List[str]
) -> None:
    """Draw the bounding boxes onto the original image and save it as a PNG file."""
    obj_detected_img = draw_bboxes(image, boxes, categories)
    obj_detected_img.save(output_file, "PNG")
    logger.info(f"Saved image with bounding boxes of detected objects to {output_file}")


def load_labels(file: Union[str, Path]) -> List[str]:
    """Load labels from text file."""
    with open(file, "r") as f:
        categories = [l.strip() for l in f.readlines()]
    return categories


def display_postprocessing_node_attributes(onnx_path: Union[str, Path]) -> None:
    """Display `BatchedNMS_TRT` post-processing node attribute."""
    # Known attribute values that were specified when generating the ONNX model
    # Ref: https://github.com/NVIDIA/TensorRT/tree/master/plugin/batchedNMSPlugin
    attrs = {
        "numClasses": 80,
        "scoreThreshold": 0.5,
        "iouThreshold": 0.45,
        "topK": 200,
        "keepTopK": 200,
        "shareLocation": 1,
        "backgroundLabelId": -1,  # no background
        "isNormalized": 1,
        "clipBoxes": 1,
    }

    try:
        import onnx

        model = onnx.load(onnx_path)
        node = model.graph.node[-1]
        FLOAT_TYPE = 1
        if node.op_type == "BatchedNMS_TRT":
            attrs = {
                attr.name: f"{attr.f:.2f}" if attr.type == FLOAT_TYPE else str(attr.i)
                for attr in node.attribute
            }
    except ModuleNotFoundError:
        pass

    logger.info("-" * 32)
    logger.info("BatchedNMS_TRT attributes")
    for name, value in attrs.items():
        logger.info(f"{name + ':':<20}{value}")
    logger.info("-" * 32)

    return attrs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT Demo")
    parser.add_argument(
        "onnx",
        type=str,
        help="ONNX model path",
    )
    parser.add_argument(
        "--trt",
        type=str,
        default="",
        help="TensorRT engine file path",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="office.jpg",
        help="Input image",
    )
    parser.add_argument(
        "--opt_size",
        type=int,
        nargs=2,
        metavar=("height", "width"),
        default=(576, 1024),  # currently set to the input image size
        help="Optimal/expected input image size",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="coco_labels.txt",
        help="Labels file path",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Float16 model datatype",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Model warmup",
    )
    args = parser.parse_args()
    precision = "float16" if args.fp16 else "float32"

    # Sanity check for our own exported models
    if precision not in args.onnx:
        logger.warning(
            "Selected ONNX model precision might not match specified precision"
        )

    # Model input has dynamic shape but we still need to specify an optimization
    # profile for TensorRT
    opt_size = tuple(args.opt_size)

    logger.info(f"Running TensorRT version {trt.__version__}")

    # Display post-processing attributes
    display_postprocessing_node_attributes(args.onnx)

    # Load categories
    categories = load_labels(args.labels)

    if not args.trt and args.onnx:
        args.trt = str(Path(args.onnx).with_suffix(".trt"))

    # Load image
    imraw = Image.open(args.image)
    image = convert_to_nchw(imraw, dtype=precision)
    image = image.astype(np.float32)

    if not Path(args.trt).exists():
        # NOTE: Unclear if performance is only really affected when input size is very
        # different from the optimal size specified when the TensorRT engine was built
        if opt_size != image.shape[-2:]:
            logger.warning(
                f"Engine optimal input shape {opt_size} does not match image shape {image.shape[-2:]}"
            )

        # Export TensorRT engine
        export_tensorrt_engine(
            args.onnx,
            args.trt,
            opt_size=opt_size,
            precision=precision,
        )

    # TensorRT inference
    model = YOLOXTensorRT(args.trt, precision=precision, input_shape=image.shape[-2:])
    model.warmup(n=args.warmup)
    outputs = model(image)

    # Draw output boxes on image
    output_file = Path(args.image).with_suffix(".output.png")
    draw_outputs(imraw, output_file, outputs, categories)
