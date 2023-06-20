import pandas as pd
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report

data = pd.read_csv('ISIC2018_Task3_Training_GroundTruth.csv')
# data = pd.read_csv('ISIC2018_Task3_Validation_GroundTruth.csv')
# data = pd.read_csv('ISIC2018_Task3_Test_GroundTruth.csv')

# # Copy images to Validation/Training/Test folder



# Crear las carpetas para cada clase
for label in data.columns[1:]:
    if not os.path.exists('Training/' + label.lower()):
        os.makedirs('Training/' + label.lower())

# Copiar las imágenes a las carpetas correspondientes
for index, row in data.iterrows():
    image_path = 'ISIC2018_Task3_Training_Input/' + row['image'] + '.jpg'
    if os.path.exists(image_path):
        for label in data.columns[1:]:
            if row[label] == 1:
                shutil.copy(image_path, 'Training/' + label.lower())





data_dir = "Training"
batch_size = 64
img_height = 160
img_width = 120

# # Create a data generator
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )

# Get the class labels
class_labels = os.listdir(data_dir)

# Count the number of images in each class
class_counts = {}
for label in class_labels:
    class_counts[label] = len(os.listdir(os.path.join(data_dir, label)))

print(class_counts)

# # Determine the target number of images
# target_count = max(class_counts.values())

# # Perform oversampling for each class
# for label in class_labels:
#     count = class_counts[label]
#     if count < target_count:
#         # Calculate the exact number of new images needed
#         num_new_images = target_count - count
#         # Load the images for this class
#         class_dir = os.path.join(data_dir, label)
#         temp_dir = os.path.join(data_dir, "temp")
#         os.makedirs(temp_dir, exist_ok=True)
#         generator = datagen.flow_from_directory(
#             data_dir,  # pass the main directory
#             classes=[label],  # specify the class for this generator
#             target_size=(img_height, img_width),
#             batch_size=batch_size,
#             class_mode=None,
#             save_to_dir=temp_dir,
#             save_format='jpg'
#         )
#         # Generate and save the new images
#         i = 0
#         for _ in generator:
#             i += batch_size
#             if i >= num_new_images:
#                 break
#         # Move the generated images to the class directory
#         for file in os.listdir(temp_dir):
#             shutil.move(os.path.join(temp_dir, file), class_dir)
#         shutil.rmtree(temp_dir)  # remove the temporary directory



# directories
# training_dir = "Training"
# validation_dir = "Validation"
# os.makedirs(validation_dir, exist_ok=True)

# # Get the class labels
# class_labels = os.listdir(training_dir)

# # fraction of images to move to validation directory
# validation_fraction = 0.4

# # move a fraction of images from each label to the validation directory
# for label in class_labels:
#     training_label_dir = os.path.join(training_dir, label)
#     validation_label_dir = os.path.join(validation_dir, label)
#     os.makedirs(validation_label_dir, exist_ok=True)

#     # get all the image files for this label
#     image_files = os.listdir(training_label_dir)

#     # randomize the image files
#     random.shuffle(image_files)

#     # calculate the number of validation images
#     num_validation_images = int(len(image_files) * validation_fraction)

#     # move the validation images to the validation directory
#     for i in range(num_validation_images):
#         shutil.move(os.path.join(training_label_dir, image_files[i]), validation_label_dir)
