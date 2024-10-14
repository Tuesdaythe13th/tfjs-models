# generate_adversarial.py

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import foolbox as fb

# Define command-line arguments
parser = argparse.ArgumentParser(description='Generate Adversarial Examples')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained KMeans model (.pkl)')
parser.add_argument('--test_dir', type=str, required=True, help='Path to the test image directory')
parser.add_argument('--output_dir', type=str, default='perturbed_images', help='Output directory for adversarial images')
parser.add_argument('--output_csv', type=str, default='perturbed_predictions.csv', help='CSV file for adversarial predictions')

args = parser.parse_args()

# Load the KMeans model
with open(args.model_path, 'rb') as file:
    kmeans = pickle.load(file)

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

def generate_adversarial_examples(images, image_ids, base_model, kmeans, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fmodel = fb.TensorFlowModel(base_model, bounds=(0, 1))
    attack = fb.attacks.FGSM()

    predictions = []

    for i, img in enumerate(images):
        img_tensor = tf.convert_to_tensor([img], dtype=tf.float32)
        adversarial = attack(fmodel, img_tensor, criterion=fb.criteria.Misclassification())

        if adversarial is not None:
            adversarial_np = adversarial.numpy()[0]
            # Extract features
            features = base_model.predict(np.expand_dims(adversarial_np, axis=0))[0]
            # Predict with KMeans
            label = kmeans.predict([features])[0]

            # Save adversarial image
            adversarial_image = tf.keras.preprocessing.image.array_to_img(adversarial_np)
            adversarial_filename = f"{image_ids[i]}_adv.jpg"
            adversarial_image.save(os.path.join(output_dir, adversarial_filename))

            # Append to predictions
            predictions.append({'image_id': image_ids[i], 'prediction_label': label})
        else:
            print(f"Failed to generate adversarial example for {image_ids[i]}")
            predictions.append({'image_id': image_ids[i], 'prediction_label': None})

    # Save predictions to CSV
    df = pd.DataFrame(predictions)
    df.to_csv(args.output_csv, index=False)
    print(f"Adversarial predictions saved to {args.output_csv}")

def main():
    # Load and preprocess test images
    images, image_ids = load_and_preprocess(args.test_dir)
    if len(images) == 0:
        print("No valid images found in the test directory.")
        return

    # Extract features (optional, since we're generating adversarial examples directly)
    # features = extract_features(images, base_model)

    # Generate adversarial examples and predictions
    generate_adversarial_examples(images, image_ids, base_model, kmeans, args.output_dir)

if __name__ == "__main__":
    main()
