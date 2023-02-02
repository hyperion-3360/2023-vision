import apriltag
import cv2
import argparse
import os
import json
import numpy as np

def export(avg, poses, sink):
    #file output
    if sink['type'].lower() == 'f':
        f = sink['dest']
        np.savetxt(f, avg, fmt="%10.5f")
        for p in poses:
            np.savetxt(f, p, fmt="%10.5f")

    elif sink['type'].lower() == 'n':
        print("not supported yet!")
    elif sink['type'].lower() == 'p':
        print(avg)
        print(poses)

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
            tag_pose = np.array(tag_dict['transform']).reshape((4,4))

            sz = tag_dict['size']

            estimated_pose = np.array(pose)

            estimated_pose[0][3] *= sz
            estimated_pose[1][3] *= sz
            estimated_pose[2][3] *= sz

            tag_relative_camera_pose = np.linalg.inv(estimated_pose)

            world_camera_pos = np.matmul(tag_pose, tag_relative_camera_pose)

            if gui:
                cv2.putText(frame, str(result.tag_id), (ptA[0], ptA[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return world_camera_pos

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

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    #device from which to acquire
    ap.add_argument("device", type=str, action='store', help="device to capture from" )

    #the game layout of AprilTags in json format
    ap.add_argument("-e --environment", dest='environment', default='environment.json', action='store', help="json file containing the details of the AprilTags env")

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
                tag_info = {x['id']: x for x in env_json['tags']}
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print("Something wrong with the environment file... :(")
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

    #this will work for USB web cams
    gstreamer_str = "v4l2src device={} ! video/x-raw,framerate=30/1,width={},height={} ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True".format(args['device'], width, height)

    #using gstreamer provides greater control over capture parameters and is easier to test the camera setup using gst-launch
    cap = cv2.VideoCapture( gstreamer_str, cv2.CAP_GSTREAMER)

    if recording:
        video_out = cv2.VideoWriter(args['record'], cv2.VideoWriter_fourcc(*'MJPG'),15, (640,480))

    #use when saving images
    img_seq = 0

    #precalculate the optimal distortion matrix and crop parameters based on the image size
    dist_coeffs = np.array(camera_params['dist'])
    camera_matrix = np.array([cam_json['fx'],0, cam_json['cx'], 0, cam_json['fy'], cam_json['cy'], 0, 0, 1]).reshape((3,3))
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
    crop_x, crop_y, crop_w, crop_h = roi

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
                #first undistort and crop it
                undistorted = cv2.undistort(frame, new_camera_matrix, dist_coeffs, None, camera_matrix)
                undistorted = undistorted[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

                #convert to grayscale
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

                options = apriltag.DetectorOptions( families=env_json['tag_family'],
                                                    debug=False, 
                                                    refine_decode=True,
                                                    refine_pose=True)
                detector = apriltag.Detector(options)

                #generate detections
                results = detector.detect(gray)

                estimated_poses = []
                # loop over the AprilTag detection results
                for r in results:
                    pose = process_detection( camera_params, detector, undistorted, r, tag_info, bool(args['gui']) or recording )
                    if isinstance(pose, np.ndarray):
                        estimated_poses.append(pose)

                if estimated_poses:
                    total = np.zeros((1,3))

                    for pose in estimated_poses:
                        total += np.array([pose[0][3], pose[1][3], pose[2][3]])

                    average = total / len(estimated_poses)

                    export(average, estimated_poses, sink)

            if args['gui']:
                # show the output image after AprilTag detection
                cv2.imshow("Image", undistorted)

            if recording:
                video_out.write(undistorted) 

    cap.release()

    if recording:
        video_out.release()

    cv2.destroyAllWindows()

    clean_sink(sink)

if __name__ == '__main__':
    main()
