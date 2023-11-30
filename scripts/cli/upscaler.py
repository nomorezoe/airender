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
from diffusers import DPMSolverSDEScheduler,StableDiffusionImg2ImgPipeline,StableDiffusionLatentUpscalePipeline,DDPMScheduler,DDIMScheduler
from RealESRGAN import RealESRGAN


def get_model_path_from_pretrained(model_id):
    if("rev-animated" in model_id):
        return True
    return False

def esrgan(image_id):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=2)
    model.load_weights("../models/esrgan/RealESRGAN_x2.pth", download=False)
    image = Image.open("../../output/"  + image_id + ".png").convert('RGB')
    sr_image = model.predict(image)
    sr_image.save("../../upscaled/"+ image_id + "_upscale.png")

def img2img_upscale(image_id, denoise, steps):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    image = Image.open("../../output/"  + image_id + ".png")
    prompt = ""
    #model = "../models/deliberate_v2.safetensors"
    model = "../models/arthemycomics.safetensors"
    nprompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck, bad_prompt_version2, bad-artist, bad-hands-5, ng_deepnegative_v1_75t, easynegative"
    
    if 'prompt' in image.text:
        prompt = image.text["prompt"]
        print("prompt: "+prompt)
    else:
        print("not has propmpt")

    if 'nprompt' in image.text:
        nprompt = image.text["nprompt"]
        print("nprompt: "+nprompt)
    else:
        print("not has nprompt")

    if 'model' in image.text:
        #model = image.text["model"]
        print("model: "+model)
    else:
        print("not has model")

    esrmodel = RealESRGAN(device, scale=2)
    esrmodel.load_weights("../models/esrgan/RealESRGAN_x2.pth", download=False)
    sr_image = esrmodel.predict(image)
    #image.convert("RGB")
    #image = image.resize((new_width, new_height))

    if(get_model_path_from_pretrained(model)):
        print("from_pretrained" + model)
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, 
                                                              local_files_only=True,
                                                              #revision="fp16" if device.type == 'cuda' else "fp32",
                                                              torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
                                                              )
    else:
        print("from_single_file" + model)
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(model,
                                                                local_files_only=True,
                                                                use_safetensors=True,
                                                                revision="fp16" if device.type == 'cuda' else "fp32",
                                                                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
                                                                )
    pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    
    #pipe = pipe.to("mps")
    #pipe.enable_attention_slicing()
    pipe.to(device) 
    pipe.enable_model_cpu_offload()

    images = pipe(prompt=prompt, negative_prompt=nprompt, num_inference_steps = steps, image=sr_image, strength=denoise, guidance_scale=7).images
    images[0].save("../../upscaled/"+ image_id + "_upscale.png")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help="image name")
    parser.add_argument('--denoise', '-d', type=float, default = 0.5, help="denoising")
    parser.add_argument('--steps', '-s', type=int, default = 30, help="steps")
    return parser.parse_args()


if __name__ == "__main__":

    torch.backends.cuda.matmul.allow_tf32 = True

    args = parse_args()
    print('arg_image_id: ' + args.image)
    print('arg_denoise: ' + str(args.denoise))
    print('arg_steps: ' + str(args.steps))

    #if (args.node == 1):
    mydir = os.getcwd()
    mydir_tmp = mydir + "/../scripts/cli"
    mydir_new = os.chdir(mydir_tmp)

    img2img_upscale(args.image, args.denoise, args.steps)
    #esrgan(args.image)
