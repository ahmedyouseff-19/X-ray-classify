from flask import Flask, request, render_template, url_for, jsonify
from PIL import Image
import numpy as np
###################################
IMG_DIM=224
############################
app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    #image_arr.shape = (1, 150, 150, 3)
    return image_arr


################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten,Dropout,GlobalAvgPool2D

model = Sequential()
model.add(Conv2D(16,3, activation='relu', input_shape=(224,224,3)))
model.add(MaxPool2D(2))
model.add(Conv2D(32, 3, activation='relu'))
model.add(Conv2D(32, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(Conv2D(64, 3, activation='relu'))
model.add(MaxPool2D(2))
model.add(GlobalAvgPool2D())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
MODEL_PATH = 'saved-bst/xray182-best'
############################################################################3

print(MODEL_PATH)
model.load_weights(MODEL_PATH)
#####################################################################

@app.route('/')
def index():

    return render_template('index.html', appName="X-Ray classify")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")

        result = model.predict(np.expand_dims(image_arr, 0))[0][0]
        print("Model predicted")

        class_name = 'abnormal' if result >= 0.5 else 'Normal'
        print(result)
        return jsonify({'prediction': class_name})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(np.expand_dims(image_arr, 0))[0][0]
        print("Model predicted")

        class_name = 'abnormal' if result >= 0.5 else 'Normal'
        print(result)

        return render_template('index.html', prediction=class_name, image='static/IMG/', appName="X-Ray classify")
    else:
        return render_template('index.html',appName="X-Ray classify")


if __name__ == '__main__':
    app.run(debug=True)