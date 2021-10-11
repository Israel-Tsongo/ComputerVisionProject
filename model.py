
from tensorflow.keras import  models
import tensorflow as tf
import cv2

import numpy as np
from base64 import b64encode



classes=["Avion","Automobile","Oiseau","Chat","Chevreuil","Chien","Grenouille","Cheval","Bateau","Camion"]


def display(path):
      image=cv2.imread(path)
           
      ret,jpeg = cv2.imencode(".jpg",image)
      print("*****",ret)
      value = b64encode(jpeg.tobytes()).decode("utf-8")  
      return value

def resize_and_normalize_img(img):    

        
        
      ####  loading image ########
      
      image = tf.io.read_file(img, name=None)
      
      #### decode the image #######

      image = tf.io.decode_jpeg(image, channels=3)
      

      #### reshape the image  && resize ######     
      
      image.set_shape([None, None, 3])
     
      image= tf.image.resize(image, (32, 32))
      
      #### convert to numpy array ######
      
      image = tf.keras.preprocessing.image.img_to_array(image)      
      
      image = np.expand_dims(image, 0)
       
		

      return image

def predict(image):
        
      image_to_classify =resize_and_normalize_img(image)
      classifierModel= models.load_model("./classifierModel/classifierModel.kerasave")  
      
            
      ##### Prediction  #####
      
      y_pred=classifierModel.predict(image_to_classify)

      print("il s'agit d'un(e) ",classes[np.argmax(y_pred)])

      return classes[np.argmax(y_pred)]
