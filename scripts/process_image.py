import torch
from model import Model
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from utils import randomize_seed_fn

import torch
from diffusers import StableDiffusionInpaintPipeline,AutoencoderKL
import time
from PIL import Image
from contextlib import contextmanager
import torch

from adetailer.common import PredictOutput
from adetailer.mask import filter_by_ratio, mask_preprocess, sort_bboxes
from adetailer import ultralytics_predict
import masking

import numpy as np
import cv2
import images
import os

def progress(step, timestep, latents):
    print(step, timestep, latents[0][0][0][0], flush=False)


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


def inpaint_it(pipeline, image, type, device):
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
    

    with change_torch_load():
        pred = predictor(ad_models[type], image, 0.3, device)

    masks = pred_preprocessing(pred)

    for mask in masks:
        mask_image=mask

        mask_blur=4
        np_mask = np.array(mask_image)
        kernel_size = 2 * int(4 * 4 + 0.5) + 1
        np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur)
        np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur)
        mask_image = Image.fromarray(np_mask)  

        inpaint_full_res_padding=32
        mask_image = mask_image.convert('L')
        crop_region = masking.get_crop_region(np.array(mask_image), inpaint_full_res_padding)
        crop_region = masking.expand_crop_region(crop_region, 840, 560, mask_image.width, mask_image.height)

        init_image = image.crop(crop_region)   
        mask_image = mask_image.crop(crop_region)  

        original_w = init_image.width
        original_h = init_image.height 
        
        init_image = images.resize_image(2, init_image, 840, 560)
        mask_image = images.resize_image(2, mask_image, 840, 560)

        #init_image.show()
        #mask_image.show()

        
        prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
        n_prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck"
    #,
        result = pipeline(prompt=prompt, width=840, height=560, negative_prompt=n_prompt, image=init_image, strength=0.4, mask_image=mask_image,  num_inference_steps=28, callback=progress).images[0] 
        #result.show()
        result = images.resize_image(1, result, original_w, original_h)

        image.paste(result, crop_region)
        #return image

    return image

def start_inpaint_pipeline(image, start_device):
    # inpaint pipo
    pipeline = StableDiffusionInpaintPipeline.from_single_file(
        "models/deliberate_v3-inpainting.safetensors",
        use_safetensors=True,
        torch_dtype=torch.float16 if start_device.type == 'cuda' else torch.float32,
        load_safety_checker=False,
        local_files_only=True
    )

    pipeline.to('cuda' if start_device.type == 'cuda' else 'mps')

    image = inpaint_it(pipeline, image, "face_yolov8n.pt", start_device)
    image = inpaint_it(pipeline, image, "hand_yolov8n.pt", start_device)
    return image


def main():

    prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
    n_prompt = "bad_prompt_version2, bad-artist, bad-hands-5, ng_deepnegative_v1_75t, easynegative"
    
    start_time = time.time()
    start_device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = Model(task_name='depth', device=start_device,
                base_model_id='models/deliberate_v3.safetensors',
                clip_skip=2)
    
    #lora
    model.pipe.load_lora_weights("./models/lora", weight_name="Drawing.safetensors")

    #vae
    vae = AutoencoderKL.from_single_file("models/vae/vae-ft-mse-840000-ema-pruned.safetensors", 
                                        local_files_only=True,
                                        use_safetensors=True, 
                                        torch_dtype=torch.float16 if start_device.type == 'cuda' else torch.float32)
    vae.to('cuda' if start_device.type == 'cuda' else 'mps')
    model.pipe.vae = vae

    # negative embedding
    model.pipe.load_textual_inversion("models/negative_embeddings/bad_prompt_version2.pt", token="bad_prompt_version2",
                                      local_files_only=True,)
    #model.pipe.load_textual_inversion("models/negative_embeddings/bad-artist-anime.pt", token="bad-artist-anime",
    #                                local_files_only=True,)
    model.pipe.load_textual_inversion("models/negative_embeddings/bad-artist.pt", token="bad-artist",
                                    local_files_only=True,)
    model.pipe.load_textual_inversion("models/negative_embeddings/bad-hands-5.pt", token="bad-hands-5",
                                    local_files_only=True,)
    model.pipe.load_textual_inversion("models/negative_embeddings/ng_deepnegative_v1_75t.pt", token="ng_deepnegative_v1_75t",
                                    local_files_only=True,)
    model.pipe.load_textual_inversion("models/negative_embeddings/easynegative.safetensors", token="easynegative",
                                    local_files_only=True,)
    
    # model.set_base_model('SdValar/deliberate2')
    # model.set_base_model('stablediffusionapi/deliberate-v2')
    # demo = create_demo(model.process_depth)

    image = Image.open("./test/testwriteimage.png")
    print(f"time - create: {time.time() - start_time}")

    re = model.process_depth(image, prompt=prompt, num_images=1, additional_prompt=None, negative_prompt=n_prompt, image_resolution=840, preprocess_resolution=512,
                            num_steps=25, guidance_scale=7.0, seed=randomize_seed_fn(seed=0, randomize_seed=True), preprocessor_name='Midas', callback=progress)
    print(f"time -1: {time.time() - start_time}")

    image = re[1]

    #image = start_inpaint_pipeline(image, start_device)
    print(f"time -2: {time.time() - start_time}")
    meta = PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("n_prompt", n_prompt)
    image.save("test.png", format="PNG", pnginfo=meta)


if __name__ == "__main__":
    main()
