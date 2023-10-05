import os

from torch.utils.data import Dataset
from torchvision.io import read_image


class DatasetMNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        if train:
            self.img_dir = './dataset/mnist/training'
        else:
            self.img_dir = './dataset/mnist/testing'

        self.transform = transform
        self.target_transform = target_transform
        self._get_img_labels()

    def __len__(self):
        return len(self.list_filenames)

    def __getitem__(self, idx):
        filename = self.list_filenames[idx]
        label = self.list_classes[idx]
        img_path = os.path.join(self.img_dir, str(label), filename)
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def _get_img_labels(self):
        self.list_filenames = []
        self.list_classes = []

        for class_i in os.listdir(self.img_dir):
            for filename_j in os.listdir(self.img_dir + "/" + class_i):
                self.list_filenames.append(filename_j)
                self.list_classes.append(int(class_i))