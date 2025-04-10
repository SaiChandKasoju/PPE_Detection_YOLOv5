from flask import Flask, render_template, Response, request, session, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
import cv2
from YOLO_Video import video_detection, webcam_detection
from YOLO_Image import detect_image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'yolov5key'
app.config['UPLOAD_FOLDER'] = 'static/files'

violation_detected = False

class UploadFileForm(FlaskForm):
    file = FileField("Upload Video", validators=[InputRequired()])
    submit = SubmitField("Run Detection")

class UploadImageForm(FlaskForm):
    file = FileField("Upload Image", validators=[InputRequired()])
    submit = SubmitField("Detect")

def generate_frames(path):
    global violation_detected
    for frame, alert_flag in video_detection(path):
        violation_detected = alert_flag
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        session['video_path'] = file_path
        return render_template('video_result.html')  # after upload
    return render_template('upload.html', form=form)  # show form

@app.route('/video')
def video():
    path = session.get('video_path', None)
    return Response(generate_frames(path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam_view')
def webcam_view():
    return render_template('webcam_result.html')

@app.route('/webcam_stream')
def webcam_stream():
    return Response(generate_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_webcam():
    global violation_detected
    for frame, alert_flag in webcam_detection():
        violation_detected = alert_flag
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/image', methods=['GET', 'POST'])
def image():
    form = UploadImageForm()
    result_path = None
    alert = False
    if form.validate_on_submit():
        file = form.file.data
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(file_path)
        result_path, alert = detect_image(file_path)
    return render_template('image.html', form=form, result=result_path, alert=alert)

@app.route('/alert_status')
def alert_status():
    global violation_detected
    return jsonify({'alert': violation_detected})

if __name__ == '__main__':
    app.run(debug=True)