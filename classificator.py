
import io
import base64
from PIL import Image
import numpy as np
import cv2



class Classificator:
    def __init__(self, model, image_string):
        self.classes_list = ['covid', 'normal'] #Informacao que veio do treinamento do modelo
        self.img_cols = 224
        self.img_rows = 224
        self.model = model
        self.image = self._get_data_from_request(image_string)   
    
    
    def get_classification(self): # codigo copiado e colado do treinamento do modelo
        image_processed = self._preprocessing(self.image)
        predIdxs = self.model.predict(image_processed, batch_size=1)
        predIdxs = np.argmax(predIdxs, axis=1)
        class_predicted = self.classes_list[predIdxs[0]]
        return class_predicted
    
    def _get_data_from_request(self, image_string): # codigo para decodificar arquivo que chegou no servidor
        im_binary = base64.b64decode(image_string[0])
        buf = io.BytesIO(im_binary)
        image = Image.open(buf)
        image = np.array(image)[:,:,:3]
        return image
    
    def _preprocessing(self, image): # codigo copiado e colado do treinamento do modelo
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.
        image = np.expand_dims(image, axis=0)
        return image
