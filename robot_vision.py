import argparse
import apriltag
from ai import trt_demo as trt
from pathlib import Path
from PIL import Image
import numpy as np
import threading
import json
from networktables import NetworkTables
import queue
import signal
import cv2

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

    #camera parameters as provided by the output of the calibrate_camera.py
    parser.add_argument("-c", "--config", dest='camera_config', default='camera.json', action='store', help="json file containing the camera parameters")

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
    opt_size = tuple((args.height, args.width))

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
    tag_info = dict()
    camera_params = dict()

    #let's load the environment
    with open(args.environment, 'r') as f:
        env_json = json.load(f)
        tag_info = {x['ID']: x for x in env_json['tags']}

    with open(args.camera_config, 'r') as f:
        cam_json = json.load(f)
        camera_params = {'params' : [ cam_json[x] for x in ('fx', 'fy', 'cx', 'cy')], 'dist' : cam_json['dist']}

    return tag_info, camera_params

def communication_thread(message_q):

    notified = [False]

    cond = threading.Condition()

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0]=True
            cond.notify()

    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

    table = None

    while True:

        if not notified[0]:
            with cond:
                print("Waiting")
                cond.wait(0.01)
                if notified[0]:
                    print("Connected!")
                    table = NetworkTables.getTable("SmartDashboard")

        item = message_q.get()

        if 'command' in item:
            if item['command'] == 'stop':
                break

        elif 'april_tag' in item:
            pos = item['april_tag']
            print( pos )
            if table:
                table.putNumberArray("position", pos )

        elif 'detection' in item:
            pred = item['detection']
            c, s, x1, y1, x2, y2  = pred
            print( "Detection of {} @ x1: {} y1: {} x2: {} y2: {} with confidence: {}".format(c, x1, y1, x2, y2, s))
            if table:
                table.putStringArray('detection', [c,str(s),str(x1),str(y1),str(x2),str(y2)])

def init_network_tables(args):
    msg_q = queue.Queue()
    NetworkTables.initialize(args.rio_ip)
    comm_thread = threading.Thread(target=communication_thread, args=(msg_q, ), daemon=True)

    return comm_thread, msg_q

def process_april_tag_detection( camera_params, detector, frame, result, tag_info, gui ):
    #the function assumes the camera is centered within the robot construction
    if result.tag_id in tag_info.keys() and result.hamming == 0:
        if gui:
            # extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = result.corners
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            ptA = (int(ptA[0]), int(ptA[1]))

            # draw the bounding box of the AprilTag detection
            cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

            # draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(result.center[0]), int(result.center[1]))
            cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        pose, e0, e1 = detector.detection_pose(result, camera_params['params'] )

        tag_dict = tag_info.get(result.tag_id)

        if tag_dict:
            tag_pose = np.zeros((4,4))
            rot = np.array(tag_dict['pose']['rotation'])
            tag_pose[0:3,0:3] = rot
            T = np.array([ tag_dict['pose']['translation'][x] for x in ['x', 'y', 'z']]).T
            tag_pose[0:3,3] = T
            tag_pose[3,3] = 1
            sz = 0.15

            estimated_pose = np.array(pose)
            estimated_pose[0][3] *= sz
            estimated_pose[1][3] *= sz
            estimated_pose[2][3] *= sz

            tag_relative_camera_pose = np.linalg.inv(estimated_pose)

            global_position = np.matmul(tag_pose, tag_relative_camera_pose)

            x, y , z = estimated_pose[0][3], estimated_pose[1][3], estimated_pose[2][3]

            return global_position

    return None

def compute_position( frame, camera_matrix, dist_coeffs, camera_params, tag_info ):
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)
    #convert to grayscale
    gray = cv2.cvtColor(undistorted , cv2.COLOR_BGR2GRAY)

    options = apriltag.DetectorOptions( families='tag16h5',
                                        debug=False, 
                                        refine_decode=True,
                                        refine_pose=True)
    detector = apriltag.Detector(options)

    #generate detections
    results = detector.detect(gray)

    estimated_poses = []
    # loop over the AprilTag detection results
    for r in results:
        pose = process_april_tag_detection( camera_params, detector, undistorted, r, tag_info, False )
        if isinstance(pose, np.ndarray):
            estimated_poses.append(pose)

    if estimated_poses:
        # compute average to increase precision and stability
        total = np.zeros(3,)

        for pose in estimated_poses:
            total += np.array([pose[0][3], pose[1][3], pose[2][3]])

        average = total / len(estimated_poses)

        return average, None

    return None, None

def vision_processing(kwargs):
    args = kwargs['args']
    cap = kwargs['camera']
    camera_params = kwargs['camera_params']
    tag_info = kwargs['tag_info']
    msg_q = kwargs['comm_msg_q']
    ai_model = kwargs['model']

    dist_coeffs = np.array(camera_params['dist'])
    fc = camera_params['params']
    camera_matrix = np.array([fc[0],0, fc[2], 0, fc[1], fc[3], 0, 0, 1]).reshape((3,3))
    precision = "float16" if args.fp16 else "float32"

    while( cap.isOpened() and not kwargs['quit'] ):
        #read a frame
        ret, frame = cap.read()

        #if we have a good frame from the camera
        if ret:
            pos, rot_mat = compute_position( frame, camera_matrix, dist_coeffs, camera_params, tag_info )

            if pos is not None:
                msg_q.put({'april_tag':(pos)})

            if args.gui:
                # show the output image after AprilTag detection
                cv2.imshow("Image", frame)
                cv2.waitKey(10)

        image = trt.convert_to_nchw(frame, dtype=precision)
        image = image.astype(np.float32)
        predictions = ai_model(image)
        for p in predictions:
            c, s, x1, y1, x2, y2  = p
            x1 *= args.width
            x2 *= args.width
            y1 *= args.height
            y2 *= args.height
            cat = kwargs['categories'][int(c)]
            msg_q.put({'detection':(cat,s,x1,y1,x2,y2)})

    if args.gui:
        cv2.destroyAllWindows()

    msg_q.put({'command':'stop'})

def setup_capture(dev, w, h):
    cap = cv2.VideoCapture( dev )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap

def main():
    kwargs = {}
    kwargs['quit'] = False

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
        kwargs['tag_info'],kwargs['camera_params'] = init_april_tag(args)

    kwargs['camera'] = setup_capture(args.device, args.width, args.height)

    comm_thread, kwargs['comm_msg_q'] = init_network_tables(args)

    comm_thread.start()

    vision_processing(kwargs)

    comm_thread.join()

#--------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
