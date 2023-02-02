gst-launch-1.0 -v udpsrc port=5000 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! ximagesink sync=false

