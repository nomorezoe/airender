import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import gc

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
from diffusers import ControlNetModel,StableDiffusionXLControlNetPipeline, DPMSolverSDEScheduler,StableDiffusionImg2ImgPipeline,StableDiffusionLatentUpscalePipeline,DDPMScheduler,DDIMScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from model import CONTROLNET_MODEL_XL_IDS
from preprocessor import Preprocessor
from images import resize_image
from diffusers.utils.testing_utils import load_image

def multi_controlnet(image_id,prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    image = Image.open("../../capture/" + image_id + ".png")
    #image = resize_image(2, image, 768, 512)
    depth_model_id = CONTROLNET_MODEL_XL_IDS["depth"]
    depth_controlnet = ControlNetModel.from_pretrained(depth_model_id,
                                                        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                                                        #local_files_only=Tru
                                                        )
    
    openpose_model_id = CONTROLNET_MODEL_XL_IDS["Openpose"]
    print(openpose_model_id)
    openpose_controlnet = ControlNetModel.from_pretrained(openpose_model_id,
                                                        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                                                        #local_files_only=True
                                                        )
    #MultiControlNetModel mcontrolnet = MultiControlNetModel([controlnet1, controlnet2])
    
    pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                "../models/dynavisionXLAllInOneStylized_Bakedvae.safetensors",
                safety_checker = None,
                use_safetensors=True, 
                controlnet = [depth_controlnet, openpose_controlnet],#, openpose_controlnet
                controlnet_conditioning_scale = [0.5,1.0],#, 1.0
                local_files_only=True,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32 
    )     
    pipe.enable_xformers_memory_efficient_attention()
    pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    pipe.to(device)
    torch.cuda.empty_cache()
    gc.collect()

    resolution = 512
    #resolution = min(MAX_IMAGE_RESOLUTION, resolution)
    print("resolution: "+ str(resolution))
    pose_preprocessor = Preprocessor()
    pose_preprocessor.load("Openpose")
    pose_control_image = pose_preprocessor(
                image=image,
                image_resolution=resolution,
                detect_resolution=resolution,
                hand_and_face=True,
            )
    #pose_control_image.show()

    depth_preprocessor = Preprocessor()
    depth_preprocessor.load("Midas")
    depth_control_image = depth_preprocessor(
                image=image,
                image_resolution=resolution,
                detect_resolution=resolution,
            )
    #depth_control_image.show()
    #pipe.enable_model_cpu_offload()
    
    negative_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
    seed=randomize_seed_fn(seed=0, randomize_seed=True)
    generator = torch.Generator().manual_seed(seed)
    results = pipe(prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            generator=generator,
            #callback=callback,
            #callback_steps = 1,
            image=[depth_control_image, pose_control_image]).images #, pose_control_image

    results[0].save("../../output/" + image_id +"_multicontrol.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help="image name")
    parser.add_argument('--prompt', '-p', type=str, help="prompt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mydir = os.getcwd()
    mydir_tmp = mydir + "/../scripts/cli"
    mydir_new = os.chdir(mydir_tmp)

    multi_controlnet(args.image, args.prompt)