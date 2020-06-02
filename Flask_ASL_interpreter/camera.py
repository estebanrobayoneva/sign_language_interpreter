import cv2
from model import AslModel
import numpy as np
from keras.models import model_from_json
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor = 0.6
model = AslModel("model.json", "model_weights.h5")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        #img_gray = img_gray[90:210, 30:160]
        img_gray = img_gray[120:240, 30:160]
        img_gray = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        cv2.imwrite('pp.jpg', img_gray)
        img_gray = img_gray[np.newaxis, :, :, np.newaxis]


        pred = AslModel.predict_asl(model, img=img_gray)
        cv2.putText(image, pred, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2 )

        x0 = 30
        y0 = 120
        #cv2.rectangle(image, (x0, y0), (x0 + 120, y0 + 120), (0, 255, 0), 2)
        cv2.rectangle(image, (x0, y0), (x0 + 120, y0 + 120), (0, 255, 0), 2)
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()