from model import download_all_controlnet_weights
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
#download_all_controlnet_weights()

pipeline = StableDiffusionPipeline.from_pretrained("stablediffusionapi/rev-animated")
#pipeline = StableDiffusionInpaintPipeline.from_pretrained("5w4n/deliberate-v2-inpainting")