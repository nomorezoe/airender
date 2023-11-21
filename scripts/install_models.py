from model import download_all_controlnet_weights
from diffusers import StableDiffusionPipeline,StableDiffusionInpaintPipeline
#download_all_controlnet_weights()
from transformers import CLIPTokenizer

CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", pad_token="!")
#pipeline = StableDiffusionPipeline.from_pretrained("stablediffusionapi/rev-animated")
#pipeline = StableDiffusionInpaintPipeline.from_pretrained("5w4n/deliberate-v2-inpainting")