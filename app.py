from flask import Flask, render_template, request
import pickle
import numpy as np
import cv2

model = pickle.load(open('faceMask_model.pkl', 'rb'))

COUNT = 0
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1
IMG_SIZE = 200

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def resultForm():
    
    global COUNT
    img = request.files['image']

    img.save('static/{}.jpg'.format(COUNT))    
    image = cv2.imread('static/{}.jpg'.format(COUNT))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(1, -1)
    pred = model.predict(img)
    print(pred)
    return render_template('result.html', data=pred[0])

if __name__ == "__main__":
    app.run(port=3000,debug=True)