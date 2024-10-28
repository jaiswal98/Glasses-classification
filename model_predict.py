from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from tensorflow.math import confusion_matrix

# Define paths
test_dir = r"C:\Rahul\AI\Dataset\archive\non\test"
glasses_folder = r"C:\Rahul\AI\Dataset\archive\non\glasses_detected"  # New folder to move glasses images
best_model = load_model('glasses_classifier.h5')

# Create the new folder if it doesn't exist
if not os.path.exists(glasses_folder):
    os.makedirs(glasses_folder)

# Function to dynamically resize images
def resize_image(img_path, target_size=(320, 320)):
    img = image.load_img(img_path)  # Load the image
    img = image.img_to_array(img)  # Convert image to array
    img = tf.image.resize(img, target_size)  # Resize dynamically
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Prediction function that uses dynamic resizing
def predict_classes(model, directory, target_size=(320, 320)):
    images = []
    filenames = []
    for img in os.listdir(directory):
        img_path = os.path.join(directory, img)
        resized_img = resize_image(img_path, target_size)  # Use resize_image function
        images.append(resized_img)
        filenames.append(img_path)
    
    images = np.vstack(images)  # Stack images for batch prediction
    predictions = model.predict(images)
    predicted_classes = (predictions > 0.5).astype('int')
    
    return filenames, predicted_classes

# Get predictions
test_image_paths, test_predictions = predict_classes(best_model, test_dir)

# Create DataFrame for the results
predictions_df = pd.DataFrame({'image_path': test_image_paths, 'predicted_class': test_predictions.flatten()})

# Print predictions
print(predictions_df)

# Move images predicted as "glasses" to the new folder
for i in range(len(predictions_df)):
    img_path = predictions_df.loc[i, 'image_path']
    predicted_class = predictions_df.loc[i, 'predicted_class']
    
    # If predicted as "glasses" (1), move the image
    if predicted_class == 1:
        # Get the filename from the path
        img_filename = os.path.basename(img_path)
        # Create the destination path
        dest_path = os.path.join(glasses_folder, img_filename)
        # Move the image
        shutil.move(img_path, dest_path)
        print(f"Moved {img_filename} to {glasses_folder}")

# Ground truth labels for the test dataset (optional)
actual_classes = [1 if 'glasses' in os.path.basename(img) else 0 for img in test_image_paths]

# Create a confusion matrix
conf_matrix = confusion_matrix(actual_classes, predictions_df['predicted_class'])

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=['No Glasses', 'Glasses'],
            yticklabels=['No Glasses', 'Glasses'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Predicted)')
plt.show()

# Visualization: Display a few images with their predictions
plt.figure(figsize=(15, 10))
for i in range(min(20, len(predictions_df))):
    plt.subplot(4, 5, i + 1)
    img_path = predictions_df.loc[i, 'image_path']
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img) / 255.0
    plt.imshow(img)
    
    predicted = predictions_df.loc[i, 'predicted_class']
    color = 'green' if predicted == 1 else 'red' 
    plt.title(f"Predicted: {'Glasses' if predicted == 1 else 'No Glasses'}", color=color)
    plt.axis('off')

plt.tight_layout()
plt.show()
