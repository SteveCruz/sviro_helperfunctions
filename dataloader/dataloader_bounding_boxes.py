##############################################################################################################################################################
##############################################################################################################################################################

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

import csv
import torch
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use('seaborn')
import matplotlib.patches as patches

##############################################################################################################################################################
##############################################################################################################################################################

def my_collate_fn(batch):
    """
    Custom collate function to define how to sample a batcht.
    Necessary since number of bounding boxes per image is not constant.
    """

    # first elements are images, stack them together into a tensor
    data = torch.stack([item[0] for item in batch])

    # second elements are torch tensors of the bounding boxes
    # make them a list
    target = [item[1] for item in batch]

    return data, target

##############################################################################################################################################################

class SVIRODetection(Dataset):
    """
    Example torch dataset for SVIRO bounding boxes.
    """
    def __init__(self, root, car, split, nbr_classes):

        # main SVIRO directory containing all vehicle folders
        self.root = Path(root) 

        # save the number of classes to use
        # if we select 4, then we consider everyday objects as background
        self.nbr_classes = nbr_classes

        # get all the grayscale images of full size
        self.images = sorted(list(self.root.glob(f"{car}/{split}/grayscale_wholeImage/*.png")))
        
        # get all the bounding box text files (csv)
        self.boxes = sorted(list(self.root.glob(f"{car}/{split}/boundingBoxes_wholeImage/*.txt")))

        # we do not want to use empty sceneries for training and testing 
        # also, we might not want to use the everyday objects
        if self.nbr_classes == 4:
            self.images = [element for element in self.images if "0_0_0" not in str(element) and "_4" not in str(element)]
            self.boxes = [element for element in self.boxes if "0_0_0" not in str(element) and "_4" not in str(element)]
        else:
            self.images = [element for element in self.images if "0_0_0" not in str(element)]
            self.boxes = [element for element in self.boxes if "0_0_0" not in str(element)]

    def __len__(self):
        return len(self.images)

    def _open_csv(self, file):
        """
        Read the bounding box csv file and transform it into a tensor.
        """ 
        data = []
        with open(file, mode="r") as f:
            reader = csv.reader(f)
            for row in reader:
                data.append([int(x) for x in row])

        data = np.array(data)
        return torch.from_numpy(data)

    def __getitem__(self, idx):

        # load image (grayscale)
        image = Image.open(self.images[idx]).convert("L")
        image = TF.to_tensor(image)

        # load bounding boxes
        boxes = self._open_csv(self.boxes[idx])

        return image, boxes

##############################################################################################################################################################

def plot_image_and_bbs(image, boxes):
    """
    Plot the input image together with the corresponding bounding boxes
    """
    
    # get the image as a numpy array of dimension HxW (no channels since grayscale)
    image = image[0].cpu().numpy()

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # display the image
    ax.imshow(image, vmin=0, vmax=1, cmap="gray")
    ax.grid(False)

    # for each bb in the image
    for current_bounding_box in boxes:

        # get the bb coordinates and the class
        bb_class, bb_x1, bb_y1, bb_x2, bb_y2 = current_bounding_box

        # define color based on class
        # red is rf
        if bb_class == 1:
            color = "r"
        # blue if ff
        elif bb_class == 2:
            color = "b"
        # green is person
        elif bb_class == 3:
            color = "g"
        # green is person
        elif bb_class == 4:
            color = "c"

        # create a rectangular patch using the bottom left coordinates and height and width
        rect = patches.Rectangle(xy=(bb_x1,bb_y1), height=bb_y2-bb_y1, width=bb_x2-bb_x1, linewidth=5, edgecolor=color, facecolor='none')

        # add the patch to the axes
        ax.add_patch(rect)

        # add text with the class information
        ax.text(bb_x1 + 15, bb_y1 - 15, f'class: {bb_class}', color=color, weight='bold')

    plt.show()

##############################################################################################################################################################

if __name__ == "__main__":

    # define dataset config parameters
    root = "/data/local_data/workingdir_g02/sds/data/SVIRO/"
    car = "hilux"
    split = "train"
    nbr_classes = 4

    # get the dataset
    dataset = SVIRODetection(root=root, car=car, split=split, nbr_classes=nbr_classes)

    # create loader for the defined dataset
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=False,
        collate_fn=my_collate_fn
    )

    # for each batch
    for batch_images, batch_boxes in train_loader:

        # plot first image of batch with the bounding boxes
        plot_image_and_bbs(batch_images[0], batch_boxes[0])
