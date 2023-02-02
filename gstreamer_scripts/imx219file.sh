#gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM), width=1280, height=720, format=(string)NV12, framerate=(fraction)30/1' ! omxh264enc ! qtmux ! filesink location=imx219.mp4 -e

gst-launch-1.0 nvarguscamerasrc wbmode=0 awblock=true gainrange="7 7" ispdigitalgainrange="3 3" exposuretimerange="7000000 7000000" aelock=true ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1' ! nvvidconv ! video/x-raw,height=512,width=512,pixel-aspect-ratio=1/1 ! omxh264enc ! qtmux ! filesink location=imx219.mp4 -e

#gst-launch-1.0 nvarguscamerasrc wbmode=0 awblock=true gainrange="7 7" ispdigitalgainrange="2 2" exposuretimerange="6000000 6000000" aelock=true ! 'video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1' ! omxh264enc ! qtmux ! filesink location=imx219.mp4 -e
