from diffusers import DiffusionPipeline, StableDiffusionPipeline
from transformers import AutoModel
import torch
from diffusers.pipelines.stable_diffusion import safety_checker
from model import Model
from PIL import Image
import random
from process_inpaint import inpaint_it
    
model = Model(task_name='depth')
model.set_base_model('XpucT/Deliberate')
#model.set_base_model('stablediffusionapi/deliberate-v2')
#demo = create_demo(model.process_depth)
image = Image.open("./test/testwriteimage.png")

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0], flush=True)


prompt = "high quality photography, 3 point lighting, flash with softbox, 4k, Canon EOS R3, hdr, smooth, sharp focus, high resolution, award winning photo, 80mm, f2.8, bokeh"
n_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
re = model.process_depth(image, prompt=None, num_images=1, additional_prompt=prompt, negative_prompt=n_prompt, image_resolution=670, preprocess_resolution=512,num_steps=25, guidance_scale=7.0, seed=int(random.randrange(4294967294)), preprocessor_name='Midas', callback=progress)#

image = re[1]

image = inpaint_it(image,"face_yolov8n.pt")
image = inpaint_it(image,"hand_yolov8n.pt")
image.save("test.png")
