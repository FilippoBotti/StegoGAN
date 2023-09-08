import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


def random_transform(A,A_mask):
    resize = transforms.Resize((286, 286),transforms.InterpolationMode.BICUBIC)
    A = resize(A)
    A_mask = resize(A_mask)

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(
        A, output_size=(256, 256))
    A = TF.crop(A, i, j, h, w)
    A_mask = TF.crop(A_mask, i, j, h, w)

    # Random horizontal flipping
    if random.random() > 0.5:
        A = TF.hflip(A)
        A_mask = TF.hflip(A_mask)
    A = TF.to_tensor(A)
    A_mask = TF.to_tensor(A_mask)

    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    return A, A_mask

class TemplateDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        print("ASDFGHJKJHGFDSDFGHJKL")
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.dir_mask_A = os.path.join(opt.dataroot,  'maskA')  # create a path '/path/to/data/trainA'
        self.dir_mask_B = os.path.join(opt.dataroot,  'maskB')  # create a path '/path/to/data/trainB'
        self.A_mask_paths = sorted(make_dataset(self.dir_mask_A, opt.max_dataset_size))
        self.B_mask_paths = sorted(make_dataset(self.dir_mask_B, opt.max_dataset_size))

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.A_mask_size = len(self.A_mask_paths)  # get the size of dataset A
        self.B_mask_size = len(self.B_mask_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        A_mask_path = self.A_mask_paths[index % self.A_size]
        # print(A_path)
        # print(A_mask_path)
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]
        B_mask_path = self.B_mask_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A_mask = Image.open(A_mask_path).convert('RGB')
        B_mask = Image.open(B_mask_path).convert('RGB')




        # apply image transformation
        A, A_mask = random_transform(A_img, A_mask)
        B, B_mask  = random_transform(B_img, B_mask)
        print(self.transform_A, self.transform_B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'A_mask': A_mask, 'B_mask': B_mask,
                'A_mask_paths': A_mask_path, 'B_mask_paths': B_mask_path,}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
