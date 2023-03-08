
import threading
import queue
import sys
import time
from networktables import NetworkTables

import apriltag
import cv2
import argparse
import os
import json
import numpy as np
import math
import signal
import euler


#tag size in meter (15 centimeters, eg 6")
TAG_SIZE = 0.15

def consumer_thread(kwargs):

    notified = [False]

    cond = threading.Condition()

    def connectionListener(connected, info):
        print(info, '; Connected=%s' % connected)
        with cond:
            notified[0]=True
            cond.notify()

    NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

    with cond:
        print("Waiting")
        if not notified[0]:
            cond.wait()

    print("Connected!")

    table = NetworkTables.getTable("SmartDashboard")

    while True:
        #no prevision right now to disable object detection, but keep it in the cards just in case
        kwargs['run_object_detection'] = not table.getBoolean('teleop', False)

        item = kwargs['image_queue'].get()

        if 'command' in item:
            if item['command'] == 'stop':
                break
        elif 'april_tag' in item:
            pos, rot = item['april_tag']
            table.putNumberArray("position", pos )
            table.putNumberArray("rotation", rot )
#        elif 'object_detection' in item:
#            pos, frame = item['object_detection']
#            table.putNumberArray("position", pos )

def export(avg, frame, sink):
    #file output
    if sink['type'].lower() == 'f':
        f = sink['dest']
        np.savetxt(f, avg, fmt="%10.5f")
    elif sink['type'].lower() == 'n':
        print("not supported yet!")
    elif sink['type'].lower() == 'p':
        x, y, z = avg[0:3]
        posx = 0
        posy = frame.shape[0]
        cv2.putText(frame, "Rel( x: {:5.2f} y: {:5.2f} z:{:5.2f}".format(x, y, z), (posx, posy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

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

        angles = euler.rotation_angles(pose[0:3,0:3], 'xyz')

        tag_dict = tag_info.get(result.tag_id)

        if tag_dict:
            tag_pose = np.zeros((4,4))
            rot = np.array(tag_dict['pose']['rotation'])
            tag_pose[0:3,0:3] = rot
            T = np.array([ tag_dict['pose']['translation'][x] for x in ['x', 'y', 'z']]).T
            tag_pose[0:3,3] = T
            tag_pose[3,3] = 1
            sz = TAG_SIZE

            estimated_pose = np.array(pose)
            estimated_pose[0][3] *= sz
            estimated_pose[1][3] *= sz
            estimated_pose[2][3] *= sz

            tag_relative_camera_pose = np.linalg.inv(estimated_pose)

            global_position = np.matmul(tag_pose, tag_relative_camera_pose)[0:3,3]

            return global_position,angles

    return None

def print_thread(kwargs):

    while True:
        item = kwargs['image_queue'].get()
        if 'command' in item:
            if item['command'] == 'stop':
                break
#        elif 'detection' in item:
#            pos, frame = item['detection']
#            print( pos, end="\r", flush=True )

def setup_sink( kwargs, threads ):

    args = kwargs['args']

    if args.ipaddr:
        NetworkTables.initialize(args.ipaddr)
        threads.append(threading.Thread(target=consumer_thread, args=(kwargs, ), daemon=True))
        return True
    elif args.filesink: 
        try:
            f = open(args.filesink, 'wb') 
            #start thread
            return True
        except(FileNotFoundError):
            print("Error: cannot open output file, defaulting to console output...")
            return False
    else:
        threads.append(threading.Thread(target=print_thread, args = (kwargs, ), daemon=True))
        return True

def build_parser():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    #device from which to acquire
    ap.add_argument("device", type=int, action='store', help="device to capture from" )

    #frame width to acquire
    ap.add_argument("-w", "--width", type=int, default=640, dest='width', action='store', help="capture width from camera")

    #frame height to acquire
    ap.add_argument("--height", type=int, default=480, dest='height', action='store', help="capture height from camera")

    #the game layout of AprilTags in json format
    ap.add_argument("-e", "--environment", dest='environment', default='env.json', action='store', help="json file containing the details of the AprilTags env")

    #camera parameters as provided by the output of the calibrate_camera.py
    ap.add_argument("-c", "--camera", dest='camera', default='camera.json', action='store', help="json file containing the camera parameters")

    #do we want gui output
    ap.add_argument("-g", "--gui", dest='gui', action='store_true', help="display AR feed from camera with AprilTag detection")

    destination = ap.add_argument_group()
    #file output destination
    filedest = destination.add_mutually_exclusive_group()
    filedest.add_argument("-f", "--file", dest='filesink', action='store', help="File destination of output results")

    #IP addr of network table server output destination 
    ipv4addr = destination.add_mutually_exclusive_group()
    ipv4addr.add_argument("-i", "--ipaddr", dest='ipaddr', action='store', help="IP v4 address of the RobotRIO, to access network tables")

    #camera calibration
    group = ap.add_argument_group()
    group_x = group.add_mutually_exclusive_group()
    group_x.add_argument("-s", "--store", dest='save_images', action='store', help="folder to save calibration images")

    #record the feed in an mp4 for subsequence processing
    group_x2 = group.add_mutually_exclusive_group()
    group_x2.add_argument("-r", "--record", dest='record', action='store', help="filename to record frames in mp4 format")

    return ap

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
    ):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def setup_capture( device, width, height ):

    if device >= 0:
        if 0:
            #this will work for USB web cams
            gstreamer_str = "v4l2src device=/dev/video{} ! video/x-raw,framerate=30/1,width={},height={} ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True".format(device, width, height)

            #using gstreamer provides greater control over capture parameters and is easier to test the camera setup using gst-launch
            cap = cv2.VideoCapture( gstreamer_str, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture( device )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    else:
        cap = cv2.VideoCapture( gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    return cap


def detect_april_tags(kwargs):

    args = kwargs['args']
    cap = kwargs['cap']
    if not args.save_images:
        camera_params = kwargs['camera_params']
        tag_info = kwargs['tag_info']
    image_queue = kwargs['image_queue']

    if args.record: 
        video_out = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),15, (args.width,args.height))

    #precalculate the optimal distortion matrix and crop parameters based on the image size
    if not args.save_images:
        dist_coeffs = np.array(camera_params['dist'])
        fc = camera_params['params']
        camera_matrix = np.array([fc[0],0, fc[2], 0, fc[1], fc[3], 0, 0, 1]).reshape((3,3))
    else:
        img_seq = 0

    while( cap.isOpened() and not kwargs['quit'] ):
        time.sleep(0.10)
        #read a frame
        ret, frame = cap.read()
    
        if args.gui:
            key = chr(cv2.waitKey(5) & 0xFF) 

        #if we have a good frame from the camera
        if ret:
            if args.save_images:
                #every time space is hit we save an image in the calibration folder
                if key == ' ':
                    cv2.imwrite(os.path.join(args.save_images, 'calibration_{}.png'.format(img_seq)),frame)
                    img_seq += 1
            else:
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)
                #convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                    pose = process_april_tag_detection( camera_params, detector, frame, r, tag_info, bool(args.gui) or args.record )
                    if pose is not None:
                        estimated_poses.append(pose)

                if estimated_poses:
                    total_pos = np.zeros(3,)
                    total_euler = np.zeros(3,)

                    for position, angles in estimated_poses:
                        total_pos += position
                        total_euler += angles

                    image_queue.put({'april_tag':(total_pos / len(estimated_poses), total_euler / len(estimated_poses))})

            if args.gui:
                # show the output image after AprilTag detection
                cv2.imshow("Image", frame)

            if args.record:
                video_out.write(frame) 

    if args.record:
        video_out.release()

    if args.gui:
        cv2.destroyAllWindows()

    image_queue.put({'command':'stop'})


def main():

    parser =  build_parser()

    args = parser.parse_args()

    thread_kwargs = {}

    thread_kwargs['args'] = args

    threads = []

    #pretty print numpy
    np.set_printoptions(precision = 3, suppress = True)

    if args.save_images:
        #wouldn't make sense not to have a gui in calibration
        args.gui = True
        #in this situation we're not trying to correct the image errors as we're producing the image that will serve for calibration!
        #check if folder exist if not create it
        path = args.save_images
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        #let's load the environment
        try:
            with open(args.environment, 'r') as f:
                env_json = json.load(f)
                thread_kwargs['tag_info'] = {x['ID']: x for x in env_json['tags']}
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print(e)
            quit()

        #let's load the camera parameters
        try:
            with open(args.camera, 'r') as f:
                cam_json = json.load(f)
                thread_kwargs['camera_params'] = {'params' : [ cam_json[x] for x in ('fx', 'fy', 'cx', 'cy')], 'dist' : cam_json['dist']}
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print("Something wrong with the camera file... :(")
            quit()

    thread_kwargs['cap'] = setup_capture(args.device, args.width, args.height)

    thread_kwargs['quit'] = False

    thread_kwargs['run_object_detection'] = True #disable object detection by default

    msg_q = queue.Queue()

    thread_kwargs['image_queue'] = msg_q

    def ctrl_c_handler(signal, frame):
        thread_kwargs['quit'] = True

    signal.signal(signal.SIGINT, ctrl_c_handler)

    if setup_sink(thread_kwargs, threads):
        #start producer thread
        threads.append( threading.Thread(target=detect_april_tags, args = (thread_kwargs, ),daemon=True))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    thread_kwargs['cap'].release()
    print()

if __name__ == '__main__':
    main()
