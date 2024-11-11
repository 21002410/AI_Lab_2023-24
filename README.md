# Ex.No: 13 Learning – Facial Emotion Recognition  
### DATE:                                                                          
### REGISTER NUMBER : 212221040027
### AIM: 
To write a program to recognizing facial expression of human is of significance in certain domains.
###  Algorithm:
1. Start the program.
2. Import required Python libraries, including NumPy, Pandas, Google Colab, Gradio, and various
scikit-learn modules.
3. Mount Google Drive using Google Colab's 'drive.mount()' method to access the data file
located in Google Drive.
4. Install the Gradio library using 'pip install gradio'.
5. Load the diabetes dataset from a CSV file ('diabetes.csv') using Pandas.
6. Separate the target variable ('Outcome') from the input features and Scale the input features
using the StandardScaler from scikit-learn.
7. Create a multi-layer perceptron (MLP) classifier model using scikit-learn's 'MLPClassifier'.
8. Train the model using the training data (x_train and y_train).
9. Define a function named 'diabetes' that takes input parameters for various features and Use
the trained machine learning model to predict the outcome based on the input features.
10. Create a Gradio interface using 'gr.Interface' and Specify the function to be used to make
predictions based on user inputs.
11. Launch the Gradio web application, enabling sharing, to allow users to input their data and
get predictions regarding diabetes risk.
12. Stop the program.
### Program:
import cv2
from model import FacialExpressionModel
import numpy as np

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExpressionModel("model.json", "model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture("C:\\Users\\Faizan\\Downloads\\Emotion Detector\\presidential_debate.mp4")

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (48, 48))
            pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
#main.py
from flask import Flask, render_template, Response
from camera import VideoCamera


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
#model.py
from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15
session = tf.compat.v1.Session(config=config)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]

### Output:
<img width="515" alt="122278339-3b7e1380-cf00-11eb-94c2-94914135f696" src="https://github.com/user-attachments/assets/d87ee03c-9945-41d8-a867-0f9eb83a3958">
<img width="696" alt="122278507-66686780-cf00-11eb-9836-f866356bcac9" src="https://github.com/user-attachments/assets/9328d70c-526c-40e9-902a-ecf0cc2b337f">





### Result:
Thus the recognizing facial expression of human is of significance executed successfully.
