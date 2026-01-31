import tensorflow as tf
import warnings
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras import models,layers
warnings.filterwarnings('ignore')
width=224
height=224
batch_size=3
data_dir=r'C:\Users\kumar\Desktop\Projects\SAT-SIRS\New Plant Diseases Dataset(Augmented)\train'
train_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(height,width),
    batch_size=batch_size
)
val_ds=tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(height,width),
    batch_size=batch_size
)
classes=train_ds.class_names
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)
train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(len(classes), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_ds, validation_data=val_ds, epochs=5)
model.save('mobilenet_model.h5')