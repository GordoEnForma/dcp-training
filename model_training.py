import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

tf.config.list_physical_devices('GPU')

img_width, img_height = 300, 300
train_data_dir = 'Dataset'
validation_data_dir = 'Test'
nb_train_samples = 10016
nb_validation_samples = 1511
batch_size = 64
epochs = 25

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=20,
                                   brightness_range=[0.8, 1.2],
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)
# base_model.summary()


nb_classes = 7

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(nb_classes, activation='softmax')(x)
 
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
model.summary()


for layer in base_model.layers:
    layer.trainable = False
# Descongelar las últimas 30 capas
for layer in base_model.layers[-30:]:
    layer.trainable = True


adam = Adam(lr=0.0001)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tfa.metrics.F1Score(num_classes=nb_classes, average='macro')])
model.summary()


# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

checkpointer = ModelCheckpoint(filepath='model.hdf5', verbose=1, save_best_only=True, monitor='val_loss')
callbacks = [checkpointer]

model.fit(train_generator,
          steps_per_epoch=nb_train_samples//batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=nb_validation_samples//batch_size,
          callbacks=callbacks)

model.save('model_gozu.h5')