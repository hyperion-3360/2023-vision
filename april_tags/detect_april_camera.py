import apriltag
import cv2
import argparse
import os
import json
import numpy as np

def process_detection( camera_params, detector, frame, result, tag_info ):
#TODO :need to use the distortion matrix (now that it is non 0!!!, camera_params should be a dict containing the list of fx,fy,cx,cy and the distortion matrix)
#the function assumes the camera is centered within the robot construction
    if result.tag_id in tag_info.keys() and result.hamming == 0:
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

        pose, e0, e1 = detector.detection_pose(result, camera_params )

        tag_dict = tag_info.get(result.tag_id)

        if tag_dict:
            tag_pose = np.array(tag_dict['transform']).reshape((4,4))
            print(tag_pose)

            sz = tag_dict['size']

            estimated_pose = np.array(pose)

            estimated_pose[0][3] *= sz
            estimated_pose[1][3] *= sz
            estimated_pose[2][3] *= sz

            tag_relative_camera_pose = np.linalg.inv(estimated_pose)

            world_camera_pos = np.matmul(tag_pose, tag_relative_camera_pose)

            cv2.putText(frame, str(result.tag_id), (ptA[0], ptA[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return world_camera_pos

    return None

def main():
    recording = False
    save_images = False

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    #device from which to acquire
    ap.add_argument("device", type=str, action='store', help="device to capture from" )

    #camera calibration
    ap.add_argument("-s --store", dest='save_images', action='store', help="folder to save calibration images")

    #record the feed in an mp4 for subsequence processing
    ap.add_argument("-r --record", dest='record', action='store', help="filename to record frames in mp4 format")

    #the game layout of AprilTags in json format
    ap.add_argument("-e --environment", dest='environment', default='environment.json', action='store', help="json file containing the details of the AprilTags env")

    #camera parameters as provided by the output of the calibrate_camera.py
    ap.add_argument("-c --camera", dest='camera', default='camera.json', action='store', help="json file containing the camera parameters")

    args = vars(ap.parse_args())

    recording = bool('record' in args.keys() and args['record'])

    save_images = bool('save_images' in args.keys() and args['save_images'])

    if save_images:
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
                camera_params = (cam_json['fx'], cam_json['fy'], cam_json['cx'], cam_json['cy'] )
        except(FileNotFoundError, json.JSONDecodeError) as e:
            print("Something wrong with the camera file... :(")
            quit()

    #this will work for USB web cams
    gstreamer_str = "v4l2src device={} ! video/x-raw,framerate=30/1,width=640,height=480! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=True".format(args['device'])

    #using gstreamer provides greater control over capture parameters and is easier to test the camera setup using gst-launch
    cap = cv2.VideoCapture( gstreamer_str, cv2.CAP_GSTREAMER)

    if recording:
        video_out = cv2.VideoWriter(args['record'], cv2.VideoWriter_fourcc(*'MJPG'),15, (640,480))

    img_seq = 0

    while( cap.isOpened() ):
        ret, frame = cap.read()

        key = cv2.waitKey(25) & 0xFF

        if key == ord('q'):
            break

        if ret:
            if save_images:
                if key == ord(' '):
                    cv2.imwrite(os.path.join(path, 'calibration_{}.png'.format(img_seq)),frame)
                    img_seq += 1
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                options = apriltag.DetectorOptions( families=env_json['tag_family'],
                                                    debug=False, 
                                                    refine_decode=True,
                                                    refine_pose=True)
                detector = apriltag.Detector(options)
                results = detector.detect(gray)

                estimated_poses = []
                # loop over the AprilTag detection results
                for r in results:
                    pose = process_detection( camera_params, detector, frame, r, tag_info )
                    if isinstance(pose, np.ndarray):
                        estimated_poses.append(pose)

                if estimated_poses:
                    total = np.zeros((1,3))

                    for pose in estimated_poses:
                        total += np.array([pose[0][3], pose[1][3], pose[2][3]])

                    average = total / len(estimated_poses)

                    # will need to be sent over to network tables
                    print(average)

            # show the output image after AprilTag detection
            cv2.imshow("Image", frame)

            if recording:
                video_out.write(frame) 

    cap.release()

    if recording:
        video_out.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
