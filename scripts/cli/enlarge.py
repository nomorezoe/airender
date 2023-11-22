import os
import sys
import gc
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
from diffusers import StableDiffusionUpscalePipeline,StableDiffusionLatentUpscalePipeline,DDPMScheduler,DDIMScheduler

def main(image_id, prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    
    model_id = "stabilityai/stable-diffusion-x4-upscaler"
    #model_id = "stabilityai/sd-x2-latent-upscaler"
    '''
    pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    )
    '''
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    )
    pipeline.to('cuda' if device.type == 'cuda' else 'mps')
    generator = torch.Generator(device="cuda").manual_seed(0)

    image = Image.open("../../output/" + image_id + ".png").convert("RGB").resize((128, 128))
    torch.cuda.empty_cache()
    gc.collect()
    upscaled_image = pipeline(prompt=prompt, 
                              image=image, 
                              num_inference_steps=20,
                              guidance_scale=0,
                              generator=generator,
                              ).images[0]
    torch.cuda.empty_cache()
    gc.collect()
    upscaled_image.save("../../output/" + image_id + "upscaled.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help="image name")
    parser.add_argument('--prompt', '-p', type=str, help="prompt")
    parser.add_argument('--node', '-n', type=int, default=1, help="prompt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('arg_image_id: ' + args.image)
    print('prompt: ' + args.prompt)

    if (args.node == 1):
        mydir = os.getcwd()
        mydir_tmp = mydir + "/../scripts/cli"
        mydir_new = os.chdir(mydir_tmp)


    main(args.image, args.prompt)
