from PIL import Image
import matplotlib.pyplot as plt
from numpy import copyto
import numpy as np
import struct
import cv2
import os
import math
import json
import random
import copy
import time
import datetime
import shutil
import sys

from abc import *
from shutil import copyfile
from distutils.dir_util import copy_tree

def _copyfileobj_patched(fsrc, fdst, length=16*1024*1024):
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        
shutil.copyfileobj = _copyfileobj_patched
