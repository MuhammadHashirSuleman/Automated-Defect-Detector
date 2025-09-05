import os
import random
import shutil

def split_dataset(base_path, split_ratio=0.2):
    image_train_path = os.path.join(base_path, 'images', 'images', 'train')
    label_train_path = os.path.join(base_path, 'labels', 'labels', 'train')
    image_val_path = os.path.join(base_path, 'images', 'images', 'val')
    label_val_path = os.path.join(base_path, 'labels', 'labels', 'val')

    os.makedirs(image_val_path, exist_ok=True)
    os.makedirs(label_val_path, exist_ok=True)

    images = [f for f in os.listdir(image_train_path) if f.endswith('.jpg')]
    random.shuffle(images)
    val_count = int(len(images) * split_ratio)
    val_images = images[:val_count]

    for img in val_images:
        shutil.move(os.path.join(image_train_path, img), os.path.join(image_val_path, img))
        label_file = img.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(label_train_path, label_file)):
            shutil.move(os.path.join(label_train_path, label_file), os.path.join(label_val_path, label_file))

if __name__ == '__main__':
    base_path = 'g:\\Ezitech Internship\\defect-detection-app\\datasets\\pipeline_dataset'
    split_dataset(base_path)