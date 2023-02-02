gst-launch-1.0 v4l2src device=/dev/video2 ! video/x-raw,framerate=30/1,width=640,height=480 ! xvimagesink
