from keras.models import model_from_json
import numpy as np

class AslModel(object):

    asl_list = ['a', 'b', 'c', 'd', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',  'u', 'v',  'w',  'x',  'y',  'z',
                '1', '2', '3', '4', '5', '7', '7', '8', '9']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()

    def predict_asl(self, img):

        label = self.loaded_model.predict_classes(img)
        pre = self.asl_list[np.argmax(label)]

        #print(label, self.asl_list[label[0]])
        return self.asl_list[label[0]]