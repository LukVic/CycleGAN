import torch
import tensorflow as tf
import os
import glob

# Input and output folder paths
input_folder = 'photo_tfrec'
output_folder = 'photo_pt'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to parse each example in the TFRecord file
def parse_example(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
    return image

# Function to process a single TFRecord file
def process_tfrecord(file_path, output_file):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_example)
    
    data = []
    for image in parsed_dataset:
        image_np = image.numpy()
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        data.append(image_tensor)
    
    # Convert the list of tensors to a PyTorch dataset
    pytorch_dataset = torch.utils.data.TensorDataset(torch.stack(data))
    
    # Save the dataset
    torch.save(pytorch_dataset, output_file)
    
    return len(data)

# Process all TFRecord files
total_images = 0
for i, tfrec_file in enumerate(glob.glob(os.path.join(input_folder, '*.tfrec'))):
    print(f"Processing {tfrec_file}...")
    output_file = os.path.join(output_folder, f'photo_dataset_{i:02d}.pt')
    images_in_file = process_tfrecord(tfrec_file, output_file)
    total_images += images_in_file
    print(f"Saved {images_in_file} images to {output_file}")

print(f"\nTotal images processed: {total_images}")
print(f"PyTorch datasets saved in {output_folder}")

# Demonstrate how to load and use the saved data
demo_file = os.path.join(output_folder, 'photo_dataset_00.pt')
loaded_dataset = torch.load(demo_file)

print(f"\nDemonstration:")
print(f"Loaded dataset from {demo_file}")
print(f"Number of images in loaded dataset: {len(loaded_dataset)}")
print(f"Shape of first image: {loaded_dataset[0][0].shape}")

# Example of how to iterate through the images
for i, (image,) in enumerate(loaded_dataset):
    if i < 5:  # Just print info for the first 5 images
        print(f"Image {i} shape: {image.shape}")
    else:
        break