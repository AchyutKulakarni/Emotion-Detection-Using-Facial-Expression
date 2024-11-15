# # app.py
# from flask import Flask, render_template, request, jsonify
# from untitled import load_model, make_prediction

# app = Flask(__name__)

# # Load the machine learning model when the app starts
# model = load_model()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get data from the request
#         data = request.get_json()

#         # Make prediction using the loaded model
#         result = make_prediction(model, data)

#         # Return the prediction result
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, render_template
# from emotion_detection import detect_emotion
# import cv2
# import base64
# import numpy as np

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion_route():
#     image_data = request.get_data()
#     image_data = base64.b64decode(image_data.split(b',')[1])
#     nparr = np.frombuffer(image_data, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     emotion_labels = detect_emotion(frame)
#     return jsonify({'emotions': emotion_labels})

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, render_template, request, jsonify
# from emotion_detection import detect_emotion
# import cv2
# import base64
# import numpy as np

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion_route():
#     try:
#         image_data = request.get_data()
#         image_data = base64.b64decode(image_data.split(b',')[1])
#         nparr = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         emotions = detect_emotion(frame)
#         return jsonify({'emotions': emotions})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# from emotion_detection import detect_emotion
# import cv2
# import base64
# import numpy as np

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion_route():
#     try:
#         # Get the image data from the request
#         image_data = request.get_json()['image']
        
#         # Decode and process the image
#         image_data = base64.b64decode(image_data.split(',')[1])
#         nparr = np.frombuffer(image_data, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         # Call the detect_emotion function
#         emotions = detect_emotion(frame)

#         # Return the result
#         return jsonify({'emotions': emotions})

#     except Exception as e:
#         return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# from keras.models import load_model
# from keras.preprocessing.image import img_to_array
# import threading

# app = Flask(__name__)
# face_classifier = cv2.CascadeClassifier(r"C:\Users\ASUS\FER model\haarcascade_frontalface_default.xml")
# emotion_classifier = load_model(r"C:\Users\ASUS\FER model\model.h5")
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# # Use threading to capture frames
# thread_lock = threading.Lock()
# emotions = []

# def detect_emotion(frame):
#     labels = []
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray)

#     for (x, y, w, h) in faces:
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

#         if np.sum([roi_gray]) != 0:
#             roi = roi_gray.astype('float') / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)

#             prediction = emotion_classifier.predict(roi)[0]
#             label = emotion_labels[prediction.argmax()]
#             labels.append(label)
#         else:
#             labels.append('No Faces')

#     return labels

# def video_stream():
#     global emotions
#     cap = cv2.VideoCapture(0)

#     while True:
#         _, frame = cap.read()
#         emotions = detect_emotion(frame)
#         with thread_lock:
#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# def emotion_stream():
#     global emotions
#     while True:
#         with thread_lock:
#             yield f'data:{", ".join(emotions)}\n\n'

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/emotion_feed')
# def emotion_feed():
#     return Response(emotion_stream(), mimetype='text/event-stream')

# if __name__ == "__main__":
#     threading.Thread(target=app.run, kwargs={'debug': True}).start()


from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)
face_classifier = cv2.CascadeClassifier(r"C:\Users\ASUS\FER model\haarcascade_frontalface_default.xml")
emotion_classifier = load_model(r"C:\Users\ASUS\FER model\model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            labels.append(label)
        else:
            labels.append('No Faces')

    return labels

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        emotions = detect_emotion(frame)
        for i, emotion in enumerate(emotions):
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)