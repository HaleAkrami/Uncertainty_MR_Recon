import os
import sys
import argparse
import numpy as np
from utils.analyze_loader import load_image_data, get_image_file_paths, normalize
from utils.subsampling import subsample
from utils.correction import correct_output
from utils.keras_parallel import multi_gpu_model
from utils.output import create_output_dir
#import subsampling, load_image_data, multi_gpu_model, get_image_file_paths, create_output_dir
from utils.constants import SLICE_WIDTH, SLICE_HEIGHT

from datetime import datetime
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from keras.datasets import mnist
from unet import UNet as UNet
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm 
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


device="cuda"
# Q for quantile regression
CONST_Q = 0.15

# Training set construction
NUM_SAMPLE_SLICES = 35

# Neural Network Parameters
RMS_WEIGHT_DECAY = .9
LEARNING_RATE = .001
FNET_ERROR_MSE = "mse"
FNET_ERROR_MAE = "mae"
FNET_ERROR_QR = 'qr'
 
# Checkpointing
CHECKPOINT_FILE_PATH_FORMAT = "fnet-{epoch:02d}.pt"
#SFX_NETWORK_CHECKPOINTS = "checkpoints"


def qr_loss(y, x, q=CONST_Q):
    custom_loss = torch.sum(torch.max(q * (y - x), (q - 1) * (y - x)))
 #   torch.sum(torch.max(Q * (recon_x - x), (Q - 1) * (recon_x - x))) #check what are x and y
    return custom_loss


    

class FNet:
    def __init__(self, num_gpus, error):
        self.architecture_exists = False
        self.num_gpus = num_gpus
        self.error = error

    def train(self, y_folded, y_original, batch_size, num_epochs, checkpoints_dir):
        """
        Trains the specialized U-net for the MRI reconstruction task

        Parameters
        ------------
        y_folded : [np.ndarray]
            A set of folded images obtained by subsampling k-space data
        y_original : [np.ndarray]
            The ground truth set of images, preprocessed by applying the inverse
            f_{cor} function and removing undersampled k-space data
        batch_size : int
            The training batch size
        num_epochs : int
            The number of training epochs
        checkpoints_dir : str
            The base directory under which to store network checkpoints 
            after each iteration
        """

        if not self.architecture_exists:
            self._create_architecture()

        
        
        X_train, X_test, X_lab_train, X_lab_test = train_test_split(
            y_folded, y_original, test_size=0.2, random_state=10003)

        X_train = np.transpose(X_train, (0, 3, 1, 2))
        X_test = np.transpose(X_test, (0, 3, 1, 2))
        X_lab_train = np.transpose(X_lab_train, (0, 3, 1, 2))
        X_lab_test = np.transpose(X_lab_test, (0, 3, 1, 2))

        train_data = TensorDataset( Tensor(X_train), Tensor(X_lab_train) )
        test_data = TensorDataset( Tensor(X_test), Tensor(X_lab_test) )

        train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              shuffle=False)
        #self.model.weight_reset()
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0

            for batch_idx, data in enumerate(tqdm(train_loader)):
                if torch.cuda.is_available():
                    in_data = data[0].cuda()
                    out_data=data[1].cuda()
                self.optimizer.zero_grad()
                recon_batch = self.model(in_data)
                loss = qr_loss(out_data, recon_batch, q=CONST_Q)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
            print(batch_idx)
            test_loss=0 
            self.model.eval()
            with torch.no_grad():
                for batch_idx, data in enumerate(test_loader):
                    if torch.cuda.is_available():
                        in_data = data[0].cuda()
                        out_data=data[1].cuda()
                    recon_batch = self.model(in_data)
                    loss = qr_loss(out_data, recon_batch, q=CONST_Q)
                    test_loss += loss.item()
                    
            
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss / len(train_loader.dataset),
            'valid_loss':test_loss / len(test_loader.dataset)
            }, checkpoints_dir+CHECKPOINT_FILE_PATH_FORMAT)
            #if batch_idx % args.log_interval == 0:
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #epoch, batch_idx * len(data), len(train_loader.dataset),
                    #100. * batch_idx / len(train_loader),
                    #loss.item() / len(data)))

            print('====> Epoch: {} Average train loss: {:.4f}'.format(
                    epoch, train_loss / len(train_loader.dataset)))
            print('====> Epoch: {} Average test loss: {:.4f}'.format(
                    epoch, test_loss / len(test_loader.dataset)))

    
    def _create_architecture(self):
        self.model = UNet(1, 1).to(device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=LEARNING_RATE, eps=1e-08,alpha=0.9, weight_decay=0) #does not have rho=RMS_WEIGHT_DECAY I used alpha=RMS_WEIGHT_DECAY
        

        #if self.num_gpus >= 2:
            #self.model = multi_gpu_model(self.model, gpus=self.num_gpus) Haleh commented

        self.architecture_exists = True


def load_and_subsample(raw_img_path, substep, low_freq_percent):
    """
    Loads and subsamples an MR image in Analyze format

    Parameters
    ------------
    raw_img_path : str
        The path to the MR image
    substep : int
        The substep to use when subsampling image slices
    low_freq_percent : float
        The percentage of low frequency data to retain when subsampling slices

    Returns
    ------------
    tuple
        A pair containing the following ordered numpy arrays:

        1. The subsampled MR image (datatype `np.float32`)
        2. The original MR image (datatype `np.float32`)
    """

    original_img = load_image_data(analyze_img_path=raw_img_path)
    subsampled_img, _ = subsample(
        analyze_img_data=original_img,
        substep=substep,
        low_freq_percent=low_freq_percent)

    original_img = np.moveaxis(original_img, -1, 0)
    original_img = np.expand_dims(original_img, -1)
    subsampled_img = np.moveaxis(np.expand_dims(subsampled_img, 3), -2, 0)

    num_slices = len(original_img)
    if num_slices > NUM_SAMPLE_SLICES:
        relevant_idx_low = (num_slices - NUM_SAMPLE_SLICES) // 2
        relevant_idx_high = relevant_idx_low + NUM_SAMPLE_SLICES
        relevant_idxs = range(relevant_idx_low, relevant_idx_high)

        subsampled_img = subsampled_img[relevant_idxs]
        original_img = original_img[relevant_idxs]

    return subsampled_img, original_img


def load_and_subsample_images(disk_path, num_imgs, substep, low_freq_percent):
    """
    Parameters
    ------------
    disk_path : str
        A path to a disk (directory) of MRI images in Analyze 7.5 format
    num_imgs : int
        The number of images to load
    substep : int
        The substep to use when each image
    low_freq_percent : float
        The percentage of low frequency data to retain when subsampling training images

    Returns
    ------------
    A tuple of training data and ground truth images, each represented
    as numpy float arrays of dimension N x 256 x 256 x 1.
    """
    file_paths = get_image_file_paths(disk_path)

    num_output_imgs = 0

    x_train = None
    y_train = None

    for i in range(len(file_paths)):
        raw_img_path = file_paths[i]

        subsampled_img, original_img = load_and_subsample(
            raw_img_path=raw_img_path,
            substep=substep,
            low_freq_percent=low_freq_percent)

        if i == 0:
            x_train = subsampled_img
            y_train = original_img
        else:
            x_train = np.vstack([x_train, subsampled_img])
            y_train = np.vstack([y_train, original_img])

        num_output_imgs += 1
        if num_output_imgs >= num_imgs:
            break

    return x_train, y_train


def main():
    parser = argparse.ArgumentParser(
        description='Train the deep neural network for reconstruction (FNet) on MR image data')
    parser.add_argument(
        '-d',
        '--disk_path',
        type=str,
        default='/ImagePTE1/ajoshi/oasis/training/RAW',
        help="The path to a disk (directory) containing Analyze-formatted MRI images"
    )
    parser.add_argument(
        '-t',
        '--training_size',
        type=int,
        default=1400,
        help="The size of the training dataset")
    parser.add_argument(
        '-e',
        '--training_error',
        type=str,
        default='qr',
        help="The type of error to use for training the reconstruction network (either 'mse' or 'mae' or 'qr')"
    )
    parser.add_argument(
        '-f',
        '--lf_percent',
        type=float,
        default=.04, # 0.04 was default
        help="The percentage of low frequency data to retain when subsampling training images"
    )
    parser.add_argument(
        '-s',
        '--substep',
        type=int,
        default=4,
        help="The substep to use when subsampling training images")
    parser.add_argument(
        '-n',
        '--num_epochs',
        type=int,
        default=50,
        help='The number of training epochs')
    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=8,  # 256,
        help='The training batch size. This will be sharded across all available GPUs'
    )
    parser.add_argument(
        '-g',
        '--num_gpus',
        type=int,
        default=1,
        help='The number of GPUs on which to train the model')
    parser.add_argument(
        '-c',
        '--checkpoints_dir',
        type=str,
        default='/ImagePTE1/akrami/MRI_Recon_results/torch/qr_' + str(CONST_Q),
        help='The base directory under which to store network checkpoints after each iteration')

    args = parser.parse_args()

    if not args.disk_path:
        raise Exception("--disk_path must be specified!")

    x_train, y_train = load_and_subsample_images(
        disk_path=args.disk_path,
        num_imgs=args.training_size,
        substep=args.substep,
        low_freq_percent=args.lf_percent)

    if len(x_train) > args.training_size:
        # Select the most relevant slices from each image
        # until the aggregate number of slices is equivalent to the
        # specified training dataset size
        training_idxs = range(args.training_size)
        x_train = x_train[training_idxs]
        y_train = y_train[training_idxs]

    net = FNet(num_gpus=args.num_gpus, error=args.training_error)
    net.train(
        y_folded=x_train,
        y_original=y_train,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        checkpoints_dir=args.checkpoints_dir)


if __name__ == "__main__":
    main()
