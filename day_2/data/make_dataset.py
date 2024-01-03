import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Get the data and process it
    train_images, train_targets = [], []    

    for i in range(6):
        train_images.append(torch.load('data/raw/train_images_{}.pt'.format(i)))
        train_targets.append(torch.load('data/raw/train_target_{}.pt'.format(i)))
    
    train_images = torch.cat(train_images, 0)
    train_images = train_images.unsqueeze(1)
    train_targets = torch.cat(train_targets, 0)
    # subtract the mean and divide by the std
    train_images = train_images.float()

    train_images = (train_images - torch.mean(train_images)) / torch.std(train_images)
    # save the processed data
    torch.save(train_images, 'data/processed/train_images.pt')
    torch.save(train_targets, 'data/processed/train_target.pt')

    # now do the same for the test data    
    test_images = torch.load('data/raw/test_images.pt')
    test_targets = torch.load('data/raw/test_target.pt')
    test_images = test_images.unsqueeze(1)
    test_images = test_images.float()
    test_images = (test_images - torch.mean(test_images)) / torch.std(test_images)
    torch.save(test_images, 'data/processed/test_images.pt')
    torch.save(test_targets, 'data/processed/test_target.pt')
  