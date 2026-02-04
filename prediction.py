import tensorflow as tf
import numpy as np
from keras.models import load_model
model=load_model(r'your path to the model')
data_dir = r'path to dataset'
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
def load_and_preprocess_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    return np.expand_dims(img_array, axis=0) / 255.0
def predict_disease(image_path):
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_label = train_ds.class_names[predicted_class]
    confidence = 100 * np.max(predictions[0])
    return predicted_label, confidence
path_input=input("Enter the path to the image:").strip('"')

print(predict_disease(path_input))
