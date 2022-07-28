import cv2
import os
import json
import copy

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms as T
#import transforms as T

from torch.optim.lr_scheduler import StepLR

from library.ai.MaskRCNN.vision.references.detection.engine import train_one_epoch, evaluate
import library.ai.MaskRCNN.vision.references.detection.utils as utils
import library.ai.MaskRCNN.vision.references.detection.transforms as T

#from detection.engine import train_one_epoch, evaluate
#import detection.utils as utils
#import detection.transforms as T