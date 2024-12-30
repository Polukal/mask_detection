import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from config import Config


class FaceMaskDataset(Dataset):
    def __init__(self, dataset_path, transform=None, train=True):
        self.dataset_path = dataset_path
        self.transform = transform
        self.train = train

        # Get all image files
        self.images = []
        self.annotations = []

        # Load all image and annotation files
        img_dir = os.path.join(dataset_path, 'images')
        ann_dir = os.path.join(dataset_path, 'annotations')

        for img_file in os.listdir(img_dir):
            if img_file.endswith('.png') or img_file.endswith('.jpg'):
                img_path = os.path.join(img_dir, img_file)
                ann_path = os.path.join(ann_dir, img_file.rsplit('.', 1)[0] + '.xml')

                if os.path.exists(ann_path):
                    self.images.append(img_path)
                    self.annotations.append(ann_path)

        # Split dataset
        split_idx = int(len(self.images) * Config.TRAIN_SPLIT)
        if train:
            self.images = self.images[:split_idx]
            self.annotations = self.annotations[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.annotations = self.annotations[split_idx:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load annotation
        ann_path = self.annotations[idx]
        boxes, labels = self._parse_annotation(ann_path)

        # Apply transformations
        if self.transform:
            transformed = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        return {
            'image': img,
            'boxes': boxes,
            'labels': labels
        }

    def _parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_idx = Config.CLASSES.index(class_name)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_idx)

        return np.array(boxes), np.array(labels)


def get_dataloader(dataset_path, batch_size, train=True):
    dataset = FaceMaskDataset(dataset_path, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)