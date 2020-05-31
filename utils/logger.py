from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import logging
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import random
from sklearn.cluster import KMeans
from sklearn.random_projection import SparseRandomProjection

import os

def get_logger(file_path):
    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


