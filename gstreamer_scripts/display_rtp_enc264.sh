gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtpjitterbuffer latency=250 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! ximagesink sync=false

