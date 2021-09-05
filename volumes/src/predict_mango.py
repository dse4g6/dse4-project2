import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_curve
from tensorflow.keras.preprocessing.image import img_to_array, load_img


# # load model
cnn_model = load_model('codecnn_mango_model.sav')

def pred_pipeline(url, cnn_model=cnn_model):
  image_url = tf.keras.utils.get_file('Mango', origin=url)
  img = tf.keras.preprocessing.image.load_img(image_url, target_size=( 224, 224 ) )
  os.remove(image_url) # Remove the cached file
  img_array = tf.keras.preprocessing.image.img_to_array(img)

  label_map = {0: 'Falan', 1: 'Kaew Kamin', 2: 'Khiaosawoey', 3: 'Manbangkhunsi'}

  predictions = cnn_model.predict(np.expand_dims(img_array, axis=0)/255)     # Vector of probabilities
  pred_labels = np.argmax(predictions, axis = 1) # We take the highest probability
  predicted_label_batch =list(map(label_map.get, pred_labels.tolist()))

  # predicted_label_batch = ""

  return predicted_label_batch