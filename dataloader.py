import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


class EmotionDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.image_col = 'image'
        self.label_col = 'emotion'

        self.data_frame[self.label_col] = self.data_frame[self.label_col].astype(
            str).str.lower()

        unique_labels = sorted(self.data_frame[self.label_col].unique())
        self.label_to_int = {label: i for i, label in enumerate(unique_labels)}
        self.int_to_label = {i: label for i, label in enumerate(unique_labels)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        img_name = self.data_frame.iloc[idx][self.image_col]
        img_path = os.path.join(self.img_dir, img_name)

        # convert img to gray scale
        image = Image.open(img_path).convert('L')

        label = self.data_frame.iloc[idx][self.label_col].lower()
        idx = self.label_to_int[label]
        # apply transform
        if self.transform:
            image = self.transform(image)

        return image, idx


if __name__ == "__main__":

    # Define Transformations (resize just in case that image size might be difference)
    data_transform = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = EmotionDataset(
        csv_file='data/legend.csv',
        img_dir='images',
        transform=data_transform
    )
    # get the amount of unique label
    print(
        f'label size(different label):{len(dataset.data_frame['emotion'].unique())}')

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    images, labels = next(iter(dataloader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")
