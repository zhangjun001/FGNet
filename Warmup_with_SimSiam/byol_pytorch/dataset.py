from __future__ import print_function, division
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class SegDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.image_paths = sorted(os.listdir(root_dir))
        self.transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        img_name = os.path.join(self.root_dir, image_name)

        image = pil_loader(img_name)

        return self.transform(image)



