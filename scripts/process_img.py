import torch
from model import Model
from PIL import Image
import random
from process_inpaint import inpaint_it
from utils import randomize_seed_fn

import torch
from diffusers import StableDiffusionInpaintPipeline
import time

start_time = time.time()  
start_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = Model(task_name='depth',device=start_device)
#model.set_base_model('SdValar/deliberate2')
#model.set_base_model('stablediffusionapi/deliberate-v2')
#demo = create_demo(model.process_depth)
image = Image.open("./test/testwriteimage.png")
print(f"time - create: {time.time() - start_time}")

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0], flush=False)


prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
n_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
re = model.process_depth(image, prompt=prompt, num_images=1, additional_prompt=None, negative_prompt=n_prompt, image_resolution=840, preprocess_resolution=512,num_steps=25, guidance_scale=7.0, seed=randomize_seed_fn(seed=0,randomize_seed=True), preprocessor_name='Midas', callback=progress)#
print(f"time -1: {time.time() - start_time}")

image = re[1]

pipeline = StableDiffusionInpaintPipeline.from_single_file(
            "models/deliberate_v3-inpainting.safetensors",
            use_safetensors=True, 
            torch_dtype=torch.float16 if start_device=="cuda" else torch.float32,
            load_safety_checker=False,
            local_files_only=True
        )
        #pipeline.load_lora_weights("./models", weight_name="Drawing.safetensors")

pipeline.to('cuda' if start_device=="cuda" else 'mps')
        #pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        #pipeline.scheduler = diffusers.DDIMScheduler.from_config(pipeline.scheduler.config)

print(f"time -3: {time.time() - start_time}")
image = inpaint_it(pipeline, image,"face_yolov8n.pt", start_device)
image = inpaint_it(pipeline, image,"hand_yolov8n.pt", start_device)
print(f"time -2: {time.time() - start_time}")
image.save("test.png")
