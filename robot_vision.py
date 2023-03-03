import argparse
from ai import trt_demo as trt
from pathlib import Path
from PIL import Image
import numpy as np
from signal import signal
import threading

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperion 3360 Chargedup 2023 vision application")

    #do we want gui output
    parser.add_argument("-g", "--gui", dest='gui', action='store_true', help="display AR feed from camera with optional AprilTag detection and or AI")

    #do we want AI
    parser.add_argument("--ai", dest='ai', action='store_true', help="enable object detection")

    #do we want AprilTag
    parser.add_argument("--apriltag", dest='apriltag', action='store_true', help="enable apriltag detection")

    #print detection results on the console
    parser.add_argument("-v", "--verbose", dest='verbose', action='store_true', help="Display detection results on the console")

    #print detection results on the console
    parser.add_argument("-i", "--ip", dest='rio_ip', type=str, default="10.33.60.2", help="RIO IP address")

    #device from which to acquire
    parser.add_argument("device", type=int, action='store', help="device to capture from" )

    #frame width to acquire
    parser.add_argument("-w", "--width", type=int, default=640, dest='width', action='store', help="capture width from camera")

    #frame height to acquire
    parser.add_argument("--height", type=int, default=480, dest='height', action='store', help="capture height from camera")

    #the game layout of AprilTags in json format
    parser.add_argument("-e", "--environment", dest='environment', default='env.json', action='store', help="json file containing the details of the AprilTags env")

    #needed when the ai is activated
    parser.add_argument( "--onnx", type=str, help="ONNX model path",)

    #speed up software starting as using precompile model
    parser.add_argument( "--trt", type=str, default="", help="TensorRT engine file path",)

    #use humand readable strings instead of index for object class
    parser.add_argument( "--labels", type=str, help="Labels file path",)

    #to specify the model is in fp16
    parser.add_argument( "--fp16", action="store_true", help="Float16 model datatype",)

    #warmup inferece to prime the pump!
    parser.add_argument( "--warmup", type=int, default=5, help="Model warmup",)

    return parser

def init_AI(args):
    precision = "float16" if args.fp16 else "float32"

    # Display post-processing attributes
    trt.display_postprocessing_node_attributes(args.onnx)

    # Load categories
    categories = trt.load_labels(args.labels)

    if not args.trt and args.onnx:
        args.trt = str(Path(args.onnx).with_suffix(".trt"))

    # Model input has dynamic shape but we still need to specify an optimization
    # profile for TensorRT
    opt_size = tuple(args.height, args.width)

    if not Path(args.trt).exists():
        # Export TensorRT engine
        trt.export_tensorrt_engine(
            args.onnx,
            args.trt,
            opt_size=opt_size,
            precision=precision,
        )

    # TensorRT inference
    model = trt.YOLOXTensorRT(args.trt, precision=precision, input_shape=[args.height, args.width])
    model.warmup(n=args.warmup)

    return model, categories

def init_april_tag(args):
    return None

def init_acquisition(args):
    return None

def init_network_tables(args):
    return None

def vision_processing(kwargs):
    return None

#--------------------------------------------------------------------------------
if __name__ == "__main__":

    kwargs = {}

    #pretty print numpy
    np.set_printoptions(precision = 3, suppress = True)

    def ctrl_c_handler(signal, frame):
        kwargs['quit'] = True

    signal.signal(signal.SIGINT, ctrl_c_handler)

    parser = build_arg_parser()

    args = parser.parse_args()

    kwargs['args'] = args

    if args.ai:
        kwargs['model'], kwargs['categories'] = init_AI(args)

    if args.apriltag:
        init_april_tag(args)

    kwargs['camera'] = init_acquisition(args)

    comm_thread, kwargs['comm_msg_q'] = init_network_tables(args)

    comm_thread.start()

    vision_processing(kwargs)

    comm_thread.join()