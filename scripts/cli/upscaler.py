import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from contextlib import contextmanager
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from model import Model
from utils import randomize_seed_fn
from adetailer.common import PredictOutput
from adetailer.mask import filter_by_ratio, mask_preprocess, sort_bboxes
from adetailer import ultralytics_predict
from settings import MAX_IMAGE_RESOLUTION
import masking
import torch
import time
import numpy as np
import cv2
import images
import argparse
from diffusers import StableDiffusionLatentUpscalePipeline,DDPMScheduler,DDIMScheduler
from RealESRGAN import RealESRGAN

def main(image_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=2)
    model.load_weights("../models/esrgan/RealESRGAN_x2.pth", download=False)
    image = Image.open("../../output/"  + image_id + ".png").convert('RGB')
    sr_image = model.predict(image)
    sr_image.save("../../output/"+ image_id + "_upscale.png")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help="image name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('arg_image_id: ' + args.image)

    #if (args.node == 1):
    mydir = os.getcwd()
    mydir_tmp = mydir + "/../scripts/cli"
    mydir_new = os.chdir(mydir_tmp)


    main(args.image)
