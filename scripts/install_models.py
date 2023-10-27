from model import download_all_controlnet_weights
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
download_all_controlnet_weights()

pipeline = StableDiffusionInpaintPipeline.from_pretrained("5w4n/deliberate-v2-inpainting")