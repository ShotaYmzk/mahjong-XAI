import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import os
import glob
import math
from tqdm import tqdm # プログレスバー表示
import logging      # ログ出力用
import sys
import time         # 時間計測用
import random       # 乱数生成用
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts # 学習率スケジューラ
import matplotlib.pyplot as plt # プロット用
from torch.amp import autocast, GradScaler # 自動混合精度 (AMP) 用
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc # ガベージコレクタ
import tqdm


