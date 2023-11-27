# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionXLPipeline,StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch
import os
import cv2
from PIL import Image

prompt = "aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"
negative_prompt = "low quality, bad quality, sketches"

# download an image
'''
image = load_image(
    "https://hf.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
)

# initialize the models and pipeline
controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float32
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
pipe = StableDiffusionXLControlNetPipeline.from_single_file(
    "../models/sd_xl_base_1.0.safetensors",
    use_safetensors=True, 
    load_safety_checker=False,
    local_files_only=True,
    controlnet=controlnet, vae=vae, torch_dtype=torch.float32
)
pipe.enable_model_cpu_offload()

# get canny image
image = np.array(image)
image = cv2.Canny(image, 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

# generate image
image = pipe(
    prompt, controlnet_conditioning_scale=controlnet_conditioning_scale, image=canny_image
).images[0]

image.save("../../output/test.png")
'''
mydir = os.getcwd()
mydir_tmp = mydir + "/../scripts/cli"
mydir_new = os.chdir(mydir_tmp)

pipeline = StableDiffusionXLPipeline.from_single_file(
    "../models/sd_xl_base_1.0.safetensors",
    use_safetensors=True, 
    load_safety_checker=False,
    local_files_only=True,torch_dtype=torch.float32)

# prompt is passed to OAI CLIP-ViT/L-14
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# prompt_2 is passed to OpenCLIP-ViT/bigG-14
prompt_2 = "Van Gogh painting"
image = pipeline(prompt=prompt, prompt_2=prompt_2, num_inference_steps=20).images[0]
image.save("../../output/test.png")