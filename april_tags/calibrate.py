from __future__ import print_function
import numpy as np
import cv2
import argparse

def main():

    parser = argparse.ArgumentParser(
        description='calibrate camera intrinsics using OpenCV')

    parser.add_argument('filenames', metavar='IMAGE', nargs='+',
                        help='input image files')

    parser.add_argument('-r', '--rows', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in vertical direction')

    parser.add_argument('-c', '--cols', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in horizontal direction')

    parser.add_argument('-s', '--size', metavar='NUM', type=float, default=1.0,
                        help='chessboard square size in user-chosen units (should not affect results)')

    parser.add_argument('-d', '--show-detections', action='store_true',
                        help='show detections in window')

    options = parser.parse_args()
    
    if options.rows < options.cols:
        patternsize = (options.cols, options.rows)
    else:
        patternsize = (options.rows, options.cols)

    sz = options.size

    x = np.arange(patternsize[0])*sz
    y = np.arange(patternsize[1])*sz

    print( sz, x, y, patternsize)

    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

    imagesize = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    win = 'Calibrate'
    cv2.namedWindow(win)

    ipoints = []

    for filename in options.filenames:

        rgb = cv2.imread(filename)
        
        if rgb is None:
            print('warning: error opening {}, skipping'.format(filename))
            continue

        cursize = (rgb.shape[1], rgb.shape[0])
        
        if imagesize is None:
            imagesize = cursize
        else:
            assert imagesize == cursize

        print('loaded ' + filename + ' of size {}x{}'.format(*imagesize))

        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        
        retval, corners = cv2.findChessboardCorners(gray, patternsize, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if retval:
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            if options.show_detections:
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                cv2.drawChessboardCorners(display, patternsize, refined, retval)
                cv2.imshow(win, display)
                while cv2.waitKey(5) not in range(128): pass
            ipoints.append( refined )
        else:
            print('warning: no chessboard found in {}, skipping'.format(filename))
    if len(ipoints) == 0:
        print('no data point to work with... :(')
    else:
        opoints = [opoints] * len(ipoints)

        retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(opoints, ipoints, imagesize, None, None )

        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        params = (fx, fy, cx, cy)

        print()
        print('all units below measured in pixels:')
        print('  fx = {}'.format(K[0,0]))
        print('  fy = {}'.format(K[1,1]))
        print('  cx = {}'.format(K[0,2]))
        print('  cy = {}'.format(K[1,2]))
        print()
        print('pastable into Python:')
        print('  fx, fy, cx, cy = {}'.format(repr(params)))
        print('json:')
        print('{')
        print('  "fx": {},'.format(K[0,0]))
        print('  "fy": {},'.format(K[1,1]))
        print('  "cx": {},'.format(K[0,2]))
        print('  "cy": {},'.format(K[1,2]))
        print('  "dist": {}'.format(dcoeffs.tolist()))
        print('}')

        print()

if __name__ == '__main__':
    main()
