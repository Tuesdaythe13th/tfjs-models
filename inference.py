# inference.py

import os
import argparse
import pandas as pd
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import numpy as np

# Define command-line arguments
parser = argparse.ArgumentParser(description='Unsupervised Extremism Detection Inference')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pkl)')
parser.add_argument('--test_dir', type=str, required=True, help='Path to the test image directory')
parser.add_argument('--output_csv', type=str, default='predictions.csv', help='Output CSV file for predictions')

args = parser.parse_args()

# Load the KMeans model
with open(args.model_path, 'rb') as file:
    model = pickle.load(file)

# Initialize ResNet50 feature extractor
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image_path, target_size=(224, 224)):
    try:
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_and_preprocess(test_dir):
    image_list = []
    image_ids = []
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(test_dir, fname)
            img = preprocess_image(path)
            if img is not None:
                image_list.append(img)
                image_ids.append(os.path.splitext(fname)[0])
            else:
                print(f"Skipping {path}")
    return np.array(image_list), image_ids

def extract_features(images, base_model):
    features = base_model.predict(images)
    return features

def generate_predictions(features, model):
    labels = model.predict(features)
    return labels

def main():
    # Load and preprocess test images
    images, image_ids = load_and_preprocess(args.test_dir)
    if len(images) == 0:
        print("No valid images found in the test directory.")
        return

    # Extract features
    features = extract_features(images, base_model)

    # Generate predictions
    labels = generate_predictions(features, model)

    # Create DataFrame and save to CSV
    predictions_df = pd.DataFrame({'image_id': image_ids, 'prediction_label': labels})
    predictions_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
