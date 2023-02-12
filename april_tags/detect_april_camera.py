
import threading
import sys
import time
from networktables import NetworkTables

if len(sys.argv) != 2:
    print("You must supply the ip address of the RoboIO in the 10.xx.xx.2 form")
    exit(0)

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0]=True
        cond.notify()

ip = sys.argv[1]
NetworkTables.initialize(server=ip)
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

print("Connected!")

table = NetworkTables.getTable("SmartDashboard")

i = 0
while True:
#    print("RobotTime:", table.getNumber("robotTime", -1))

#    table.putNumber("JetsonTime", i)
    print("JetsonTime:", table.getNumber("JetsonTime", -1))

    time.sleep(1)

    i+=1



import apriltag
import cv2
import argparse
import os
import json
import numpy as np
import math

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

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta), R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta), R[0,0]/np.cos(beta))
    return np.array([beta, alpha, gamma])

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

def process_detection( camera_params, detector, frame, result, tag_info, gui ):
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
#            abs_pos = global_position[0:3, 3].T

#            if gui:
#                cv2.putText(frame, "Rel( x: {:5.2f} y: {:5.2f} z:{:5.2f}".format(x, y, z), (ptA[0], ptA[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
#                cv2.putText(frame, "Abs( x: {:5.2f} y: {:5.2f} z:{:5.2f}".format(abs_pos[0], abs_pos[1], abs_pos[2]), (ptA[0], ptA[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            return global_position

    return None

def setup_sink( args ):
    sink = {}
    sink['type'] = 'f' if args['filesink'] else 'n' if args['nettable'] else 'p'

    if sink['type'] == 'f' :
        try:
            sink['dest'] = open(args['filesink'], 'wb') 
        except(FileNotFoundError):
            print("Error: cannot open output file, defaulting to console output...")
            sink['type'] = 'p'
    #TODO: support nework table

    return sink

def clean_sink( sink ):
    if sink['type'] == 'f':
        sink['dest'].close()

def main():
    recording = False
    save_images = False

    #pretty print numpy
    np.set_printoptions(precision = 3, suppress = True)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    #device from which to acquire
    ap.add_argument("device", type=str, action='store', help="device to capture from" )

    #the game layout of AprilTags in json format
#    ap.add_argument("-e --environment", dest='environment', default='environment.json', action='store', help="json file containing the details of the AprilTags env")
    ap.add_argument("-e --environment", dest='environment', default='env.json', action='store', help="json file containing the details of the AprilTags env")

    #camera parameters as provided by the output of the calibrate_camera.py
    ap.add_argument("-c --camera", dest='camera', default='camera.json', action='store', help="json file containing the camera parameters")

    #do we want gui output
    ap.add_argument("-g --gui", dest='gui', action='store_true', help="display AR feed from camera with AprilTag detection")

    destination = ap.add_argument_group()
    #file output destination
    filedest = destination.add_mutually_exclusive_group()
    filedest.add_argument("-f --file", dest='filesink', action='store', help="File destination of output results")

    #networktable output destination
    networktabledest = destination.add_mutually_exclusive_group()
    networktabledest.add_argument("-n --networktable", dest='nettable', action='store', help="Networktable IP:port destination of output results")

    #camera calibration
    group = ap.add_argument_group()
    group_x = group.add_mutually_exclusive_group()
    group_x.add_argument("-s --store", dest='save_images', action='store', help="folder to save calibration images")

    #record the feed in an mp4 for subsequence processing
    group_x2 = group.add_mutually_exclusive_group()
    group_x2.add_argument("-r --record", dest='record', action='store', help="filename to record frames in mp4 format")

    args = vars(ap.parse_args())

    recording = bool('record' in args.keys() and args['record'])

    save_images = bool('save_images' in args.keys() and args['save_images'])

    if save_images:
        #wouldn't make sense not to have a gui in calibration
        args['gui'] = True
        #in this situation we're not trying to correct the image errors as we're producing the image that will serve for calibration!
        #check if folder exist if not create it
        path = args['save_images']
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        #let's load the environment
        try:
            with open(args['environment'], 'r') as f:
                env_json = json.load(f)
                tag_info = {x['ID']: x for x in env_json['tags']}
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print(e)
            quit()

        try:
            with open(args['camera'], 'r') as f:
                cam_json = json.load(f)
                camera_params = {'params' : [ cam_json[x] for x in ('fx', 'fy', 'cx', 'cy')], 'dist' : cam_json['dist']}
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print("Something wrong with the camera file... :(")
            quit()

    #TODO: do we want to pass these as arguments... perhaps
    width = 640
    height = 480

    if 0:
        #this will work for USB web cams
        gstreamer_str = "v4l2src device={} ! video/x-raw,framerate=30/1,width={},height={} ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True".format(args['device'], width, height)

        #using gstreamer provides greater control over capture parameters and is easier to test the camera setup using gst-launch
        cap = cv2.VideoCapture( gstreamer_str, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture( 0 )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if recording:
        video_out = cv2.VideoWriter(args['record'], cv2.VideoWriter_fourcc(*'MJPG'),15, (640,480))

    #use when saving images
    img_seq = 0

    #precalculate the optimal distortion matrix and crop parameters based on the image size
    if not save_images:
        dist_coeffs = np.array(camera_params['dist'])
        camera_matrix = np.array([cam_json['fx'],0, cam_json['cx'], 0, cam_json['fy'], cam_json['cy'], 0, 0, 1]).reshape((3,3))

    sink = setup_sink(args)

    while( cap.isOpened() ):
        #read a frame
        ret, frame = cap.read()

        key = cv2.waitKey(5) & 0xFF

        #check if we quit
        if key == ord('q'):
            break

        #if we have a good frame from the camera
        if ret:
            if save_images:
                #every time space is hit we save an image in the calibration folder
                if key == ord(' '):
                    cv2.imwrite(os.path.join(path, 'calibration_{}.png'.format(img_seq)),frame)
                    img_seq += 1
            else:
#                frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, None)
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
                    pose = process_detection( camera_params, detector, frame, r, tag_info, bool(args['gui']) or recording )
                    if isinstance(pose, np.ndarray):
                        estimated_poses.append(pose)

                if estimated_poses:
                    total = np.zeros(3,)

                    for pose in estimated_poses:
                        total += np.array([pose[0][3], pose[1][3], pose[2][3]])

                    average = total / len(estimated_poses)

                    export(average, frame, sink)

            if args['gui']:
                # show the output image after AprilTag detection
                cv2.imshow("Image", frame)

            if recording:
                video_out.write(frame) 

    cap.release()

    if recording:
        video_out.release()

    cv2.destroyAllWindows()

    clean_sink(sink)

if __name__ == '__main__':
    main()
