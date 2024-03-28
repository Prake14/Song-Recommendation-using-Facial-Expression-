import os
import cv2
import numpy as np
import base64 
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the pre-trained model
json_file = open("model.json", "r")
loaded_json_model = json_file.read()
json_file.close()
model = model_from_json(loaded_json_model)
model.load_weights("model_weights.h5")
emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

@app.route('/')
def index():
    return render_template('index.html', predicted_emotion=None)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if the request contains an image
        if 'emotion_image' not in request.form:
            return redirect(request.url)

        # Get the captured image as a data URL
        image_data_url = request.form['emotion_image']

        # Decode the data URL to obtain the image bytes
        image_data = image_data_url.split(',')[1].encode()
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform emotion prediction on the captured image
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

        if len(faces_detected) > 0:
            (x, y, w, h) = faces_detected[0]
            roi_gray = gray_img[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img = roi_gray.reshape((1, 48, 48, 1))
            img = img / 255.0

            max_index = np.argmax(model.predict(img), axis=-1)[0]
            predicted_emotion = emotions[max_index]

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness=3)
            cv2.putText(image, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_image.jpg'), image)
            cv2.imwrite(os.path.join('static', 'predicted_image.jpg'),image)

            # Load music files based on predicted emotion
            music_folder = os.path.join('music', predicted_emotion.lower())
            music_files = os.listdir(music_folder)

            return render_template('index.html', predicted_emotion=predicted_emotion, music_files=music_files)
        else:
            return render_template('index.html', predicted_emotion="No face detected in the image.")

    return render_template('index.html', predicted_emotion=None)

if __name__ == '__main__':
    app.run(debug=True)
