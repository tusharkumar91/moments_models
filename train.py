# demo code for using the RGB model trained on Moments in Time
# load the trained model then forward pass on a given image
# By Bolei Zhou

import os
import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.models as models
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms as trn

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from action_dataset_loader import ActionDataset
from torch.utils.data import DataLoader

def load_model(modelID, categories):
    if modelID == 1:
        weight_file = 'moments_RGB_resnet50_imagenetpretrained.pth.tar'
        if not os.access(weight_file, os.W_OK):
            weight_url = 'http://moments.csail.mit.edu/moments_models/' + weight_file
            os.system('wget ' + weight_url)
        print(models.__dict__)
        exit(0)
        model = models.__dict__['resnet50'](num_classes=len(categories))
        
        useGPU = 0
        if useGPU == 1:
            checkpoint = torch.load(weight_file)
        else:
            checkpoint = torch.load(weight_file, map_location=lambda storage,
                                    loc: storage)  # allow cpu

        state_dict = {str.replace(str(k), 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    model.fc = nn.Linear(2048, 2)
    return model


def load_categories():
    """Load categories."""
    with open('category_momentsv1.txt') as f:
        return [line.rstrip() for line in f.readlines()]




def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              weight_decay,
              save_cp=True,
              img_scale=0.5):
    train_dataset = ActionDataset('../NymbleData/yt_frames_mix', '../NymbleData/yt_frames_add', split='train')
    val_dataset = ActionDataset('../NymbleData/yt_frames_mix', '../NymbleData/yt_frames_add', split='val')
    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {len(train_dataset)}
        Device:          {device.type}
    ''')
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # validate_net(net, val_loader, device)
    # exit(0)
    best_loss = np.inf
    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        with tqdm(total=len(train_dataset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for item in train_loader:
                imgs, labels = item
                imgs = imgs.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.long)
                
                pred = net(imgs)
                loss = criterion(pred, labels)
                epoch_loss.append(loss.item())
                
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                optimizer.zero_grad()
                loss.backward()
                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                
                pbar.update(imgs.shape[0])
        
        net.eval()
        val_loss = validate_net(net, val_loader, criterion, device)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Better Loss Found')
            torch.save(net.state_dict(), 'best_model.pth'.format(val_loss.item()))
        print('Epoch Mean Loss : {}'.format(np.mean(epoch_loss)))
        # if save_cp:
        #     try:
        #         os.mkdir(dir_checkpoint)
        #         logging.info('Created checkpoint directory')
        #     except OSError:
        #         pass
        #     torch.save(net.state_dict(),
        #                dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        #     logging.info(f'Checkpoint {epoch + 1} saved !')


def validate_net(model, val_loader, criterion, device):
    valid_losses = []
    with torch.no_grad():
        print('starting evaluating on validation set')
        for item in val_loader:
            imgs, labels = item
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
    
            pred = model(imgs)
            loss = criterion(pred, labels)
            valid_losses.append(loss.item())
    print('Validation loss : {}'.format(np.mean(valid_losses)))
    return np.mean(valid_losses)


def get_args():
    parser = argparse.ArgumentParser(description='Train the Network on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-w', '--weight-decay', metavar='WD', type=float, nargs='?', default=0.00005,
                        help='Weright Decay', dest='weight_decay')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a checkpoint file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    modelID = 1
    dataset = 'moments'

    # load categories
    categories = load_categories()

    # load the model
    model = load_model(modelID, categories)
    
    train_net(net=model,
              epochs=args.epochs,
              batch_size=args.batchsize,
              lr=args.lr,
              weight_decay=args.weight_decay,
              device=device,
              img_scale=args.scale)
#     except KeyboardInterrupt:
#         torch.save(net.state_dict(), 'INTERRUPTED.pth')
#         logging.info('Saved interrupt')
#         try:
#             sys.exit(0)
#         except SystemExit:
#             os._exit(0)
#
# if __name__ == '__main__':
#     modelID = 1
#     dataset = 'moments'
#
#     # load categories
#     categories = load_categories()
#
#     # load the model
#     model = load_model(modelID, categories)
#
#     import glob
#     pred = [0]* len(glob.glob('../NymbleData/frames/*.jpeg'))
#     print(len(pred))
#     from tqdm import tqdm
#     for fname in tqdm(glob.glob('../NymbleData/frames/*.jpeg')):
#         image_idx = int(fname.split('/')[-1].split('.')[0])-1
#         if image_idx > 100:
#             continue
#         img = Image.open('../NymbleData/frames/921.jpeg')
#         input_img = V(tf(img).unsqueeze(0), volatile=True)
#
#         # forward pass
#         logit = model.forward(input_img)
#         h_x = F.softmax(logit, 1).data.squeeze()
#         probs, idx = h_x.sort(0, True)
#         #print(img_url)
#         # output the prediction of action category
#         #print('--Top Actions:')
#         for i in range(0, 5):
#             #print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
#             if categories[idx[i]] == 'stirring':
#                 if probs[i] > 0.3:
#                     pred[image_idx] = 1
#                     #print(pred)
#     print(pred[:100])
