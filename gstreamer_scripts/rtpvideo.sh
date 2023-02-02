gst-launch-1.0 videotestsrc ! videoconvert ! omxh264enc !rtph264pay config-interval=1 pt=96 ! udpsink host=192.168.0.219 port=5000
