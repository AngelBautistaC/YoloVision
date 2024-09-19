from flask import Flask, Response
import cv2

app = Flask(__name__)

def gen_frames():
    """Generator function that captures frames from the webcam and yields them as JPEG-encoded images."""
    cap = cv2.VideoCapture(1)  # Capture video from the default webcam (device index 0)
    if not cap.isOpened():
        raise RuntimeError("Could not start webcam.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue  # Skip the frame if encoding failed

        frame = buffer.tobytes()
        # Yield the output frame in byte format as part of the multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the Flask app on localhost port 5000
    app.run(host='localhost', port=5000, debug=True)
