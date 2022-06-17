# import the modules
import os
import glob
from setuptools.namespaces import flatten
import random
from itertools import islice
from typing import Any
import cv2 as cv
from torchvision.datasets.vision import VisionDataset


# Create class for loding dataset with class folders as labels and images.
class CustomDataset(VisionDataset):
    """
    Constructs a dataset loader for images with in different folders.

    Parameters
    ----------
    root: str
        Path to the dataset directory.
    train: bool
        If 'True', create a train dataset.
    test: bool
        If 'True', create a test dataset.
    validation: bool
        If 'True', create a validation dataset.
    Transform:
        Provide the transforms from torchvision.

    Returns
    -------
    Images with corresponding lables.
    """
    
    def __init__(self,root:str,train=False,test=False,validation=False,transform=None):
        
        self.root = root
        self.transform = transform

        # Define train,test and validataion sizes in percentage.
        self.train_size = 0.7
        self.test_size = 0.2
        self.validation_size = 0.1

        self.class_to_idx = self.get_classes()[2]

        # train set image_paths
        if train == True:
            self.imagepaths = self.get_dataset_paths()[0]
        # test set image_paths
        if test == True:
            self.imagepaths = self.get_dataset_paths()[1]
        # Validation set image_paths
        if validation == True:
            self.imagepaths = self.get_dataset_paths()[2]

    def get_classes(self):
        """
        Create dictionary for class to index and index to class.

        Returns
        -------
        class_names: list
        index_to_class: dict
        class_to_index: dict
        """
        class_names = []
        for root_dir,directories,files in os.walk(self.root):
            for name in directories:
                class_names.append(name)
        idx_to_class = {key:value for key,value in enumerate(class_names)}
        class_to_idx = {value:key for key,value in enumerate(class_names)}
        return class_names,idx_to_class,class_to_idx

    def get_dataset_paths(self):
        """
        Get paths for the images based on the type of dataset.

        Returns
        -------
        train_image_paths: 
            List of strings for the paths of train images
        test_image_paths: 
            List of strings for the paths of test images
        validation_image_paths: 
            List of strings for the paths of validation images
        """
        imagepaths = []
        for path in glob.glob(self.root + '/*'):
            imagepaths.append(glob.glob(path + '/*'))

        image_paths = list(flatten(imagepaths))
        random.shuffle(image_paths)
        total_images = len(image_paths)

        # Splitting the dataset into train, test and validiation sets.     
        splits = [int(self.train_size * total_images),
                  int(self.test_size * total_images), 
                  int(self.validation_size * total_images)]

        output = [list(islice(image_paths,elem)) for elem in splits]

        train_image_paths = output[0]
        test_image_paths = output[1]
        valid_image_paths = output[2]
        
        return train_image_paths, test_image_paths, valid_image_paths

    # Magic function
    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.imagepaths)

    # Magic function
    def __getitem__(self, idx: int) -> Any:
        """
        Returns one training example.
        """
        image_file_path = self.imagepaths[idx]

        # Reading the image
        image = cv.imread(image_file_path)

        # Changing the default BGR version of cv2 to RGB channels.
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Creating the labels
        label = image_file_path.split("/")[-2]
        label = self.class_to_idx[label]

        # Applying transform to the image
        if self.transform:
            image = self.transform(image)
        return image, label