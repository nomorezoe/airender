from diffusers import DiffusionPipeline, StableDiffusionPipeline
from transformers import AutoModel
import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from model import Model
from PIL import Image
import random
from process_inpaint import inpaint_it
    
model = Model(task_name='depth')
#model.set_base_model('SdValar/deliberate2')
#model.set_base_model('stablediffusionapi/deliberate-v2')
#demo = create_demo(model.process_depth)
image = Image.open("./test/testwriteimage.png")

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0], flush=True)


prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
n_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
re = model.process_depth(image, prompt=prompt, num_images=1, additional_prompt=prompt, negative_prompt=n_prompt, image_resolution=840, preprocess_resolution=512,num_steps=25, guidance_scale=7.0, seed=int(random.randrange(4294967294)), preprocessor_name='Midas', callback=progress)#

image = re[1]

image = inpaint_it(image,"face_yolov8n.pt")
image = inpaint_it(image,"hand_yolov8n.pt")
image.save("test.png")
