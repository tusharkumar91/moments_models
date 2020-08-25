import json
import sys
import os

import cv2
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tf


class ActionDataset(Dataset):
    def __init__(self, mix_frames_folder, add_frames_folder, split='train'):
        super(ActionDataset, self).__init__()
        self.mix_img_files = []
        for dir_name in os.listdir(mix_frames_folder):
            for img_path in glob.glob(os.path.join(mix_frames_folder, dir_name, '*.jpeg')):
                self.mix_img_files.append(img_path)

        self.add_img_files = []
        for dir_name in os.listdir(add_frames_folder):
            for img_path in glob.glob(os.path.join(add_frames_folder, dir_name, '*.jpeg')):
                self.add_img_files.append(img_path)
        
        if split == 'train':
            self.mix_img_files = self.mix_img_files[:int(len(self.mix_img_files)*0.9)]
            self.add_img_files = self.add_img_files[:int(len(self.add_img_files) * 0.9)]
        else:
            self.mix_img_files = self.mix_img_files[:int(len(self.mix_img_files) * 0.1)]
            self.add_img_files = self.add_img_files[:int(len(self.add_img_files) * 0.1)]
        print('Total {} stirring images loaded'.format(len(self.mix_img_files)))
        print('Total {} adding images loaded'.format(len(self.add_img_files)))
        self.labels = [0]*len(self.add_img_files) + [1] * len(self.mix_img_files)
        self.images = self.add_img_files + self.mix_img_files
        label_image_mapping = list(zip(self.images, self.labels))
        np.random.shuffle(label_image_mapping)
        self.images, self.labels = zip(*label_image_mapping)
        print('Total {} images and {} labels loaded'.format(len(self.images), len(self.labels)))
        weights = [len(self.mix_img_files) / (len(self.mix_img_files)+ len(self.add_img_files)),
                   len(self.add_img_files) / (len(self.mix_img_files) + len(self.add_img_files))]
        self.weights = torch.from_numpy(np.array(weights)).float()
        print('Weights of classes : {}'.format(self.weights))
        
    def load_transform(self):
        """Load the image transformer."""
        transforms = tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transforms
    
    def __len__(self):
        return len(self.mix_img_files + self.add_img_files)
    
    def __getitem__(self, index):
        #print(self.images[index])
        img = Image.open(self.images[index])
        img = self.load_transform()(img)
        label = self.labels[index]
        return img, label


if __name__ == '__main__':
    dataset = ActionDataset('../NymbleData/yt_frames_mix', '../NymbleData/yt_frames_add', split='train')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
   #print(len(dataset))
    for idx, item in enumerate(data_loader):
        img, label = item
        print(img.shape, label)
        if idx > 5:
            exit(0)

