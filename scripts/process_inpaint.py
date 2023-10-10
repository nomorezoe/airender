from PIL import Image
from contextlib import contextmanager
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from transformers import AutoModel
import torch
from diffusers.pipelines.stable_diffusion import safety_checker
import diffusers

from adetailer.common import PredictOutput
from adetailer.mask import filter_by_ratio, mask_preprocess, sort_bboxes
from adetailer import ultralytics_predict
import masking

import numpy as np
import cv2
import images
import os

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0], flush=True)

def pred_preprocessing(pred: PredictOutput):
    pred = filter_by_ratio(
        pred, low=0.0, high=1.0
    )
    pred = sort_bboxes(pred, 0)
    return mask_preprocess(
        pred.masks,
        kernel=4,
        x_offset=0,
        y_offset=0,
        merge_invert='None',
    )

@contextmanager
def change_torch_load():
    orig = torch.load
    try:
        torch.load = torch.load#safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


def inpaint_it(image, type):
    predictor = ultralytics_predict

    ad_models = {
    "face_yolov8n.pt":
    os.path.dirname(os.path.abspath(__file__)) +'/models/adetailer/face_yolov8n.pt',
    'face_yolov8s.pt':
    os.path.dirname(os.path.abspath(__file__)) +'/models/adetailer/face_yolov8s.pt',
    'hand_yolov8n.pt':
    os.path.dirname(os.path.abspath(__file__)) +'/models/adetailer/hand_yolov8n.pt',
    'person_yolov8n-seg.pt':
    os.path.dirname(os.path.abspath(__file__)) +'/models/adetailer/person_yolov8n-seg.pt',
    'person_yolov8s-seg.pt':
    os.path.dirname(os.path.abspath(__file__)) +'/models/adetailer/person_yolov8s-seg.pt'}

#/Users/zoewang/Documents/renderbackend/ControlNet-v1-1/models/adetailer/face_yolov8n.pt

    with change_torch_load():
        pred = predictor(ad_models[type], image, 0.3, "mps")

    bboxes=pred.bboxes
    print('bboxes')
    print(bboxes)
    masks = pred_preprocessing(pred)
    #masks[0].show()

    for mask in masks:
        mask_image=mask

        mask_blur=4
        np_mask = np.array(mask_image)
        kernel_size = 2 * int(4 * 4 + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), 4)
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), 4)
        mask_image = Image.fromarray(np_mask)  

        inpaint_full_res_padding=32
        mask_image = mask_image.convert('L')
        crop_region = masking.get_crop_region(np.array(mask_image), inpaint_full_res_padding)
        crop_region = masking.expand_crop_region(crop_region, 840, 560, mask_image.width, mask_image.height)
        x1, y1, x2, y2 = crop_region  
        print('crop_region')        
        print(crop_region)      

        init_image = image.crop(crop_region)   
        mask_image = mask_image.crop(crop_region)  

        original_w = init_image.width
        original_h = init_image.height 
        
        init_image = images.resize_image(2, init_image, 840, 560)
        mask_image = images.resize_image(2, mask_image, 840, 560)

        #init_image.show()
        #mask_image.show()

        pipeline = StableDiffusionInpaintPipeline.from_single_file(
            #'runwayml/stable-diffusion-inpainting',
            #"CompVis/ldm-super-resolution-4x-openimages",
            #"stablediffusionapi/deliberate-v2",
            #"5w4n/deliberate-v2-inpainting",
            #"Uminosachi/Deliberate-inpainting",
            #"XpucT/Deliberate",
            #"stabilityai/stable-diffusion-2-inpainting",
            "models/deliberate_v3-inpainting.safetensors",
            #use_safetensors=True, 
            safety_checker=None,
            torch_dtype=torch.float32,
            load_safety_checker=False,
            local_files_only=True,
        )
        #pipeline.load_lora_weights("./models", weight_name="Drawing.safetensors")

        pipeline.to('mps')
        #pipeline.scheduler = diffusers.EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
        #pipeline.scheduler = diffusers.DDIMScheduler.from_config(pipeline.scheduler.config)

        prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
        n_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
    #,
        result = pipeline(prompt=prompt, width=840, height=560, negative_prompt=n_prompt, image=init_image, strength=0.4, mask_image=mask_image,  num_inference_steps=28, callback=progress).images[0] 
        #result.show()
        result = images.resize_image(1, result, original_w, original_h)


        #final_inpainted = Image.composite(inpainted, image, mask)
        image.paste(result, crop_region)
        #return image

    return image



