gst-launch-1.0 videotestsrc ! videoconvert ! omxh264enc ! qtmux ! filesink location=filename.mp4 -e
