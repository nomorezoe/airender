# ,ControlNetModel，StableDiffusionControlNetInpaintPipeline

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
from diffusers import EulerAncestralDiscreteScheduler,DPMSolverSDEScheduler,DPMSolverSinglestepScheduler,StableDiffusionControlNetPipeline, DPMSolverMultistepScheduler,AutoencoderKL, ControlNetModel, StableDiffusionInpaintPipeline
from model import CONTROLNET_MODEL_IDS
import gc
from preprocessor import Preprocessor

# from controlnet_aux import OpenposeDetector

def controlnet_progress(step, timestep, latents):
    print("controlnet_progress:" + str(step), flush=True)


def inpaint_progress(step, timestep, latents):
    print("inpaint_progress:" + str(step), flush=True)

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
        torch.load = torch.load  # safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


def get_inpaint_masks(image, type, device):
    predictor = ultralytics_predict
    mydir = os.getcwd()
    ad_models = {
        "face_yolov8n.pt":
        mydir + '/../models/adetailer/face_yolov8n.pt',
        'face_yolov8s.pt':
        mydir +
        '/../models/adetailer/face_yolov8s.pt',
        'hand_yolov8n.pt':
        mydir +
        '/../models/adetailer/hand_yolov8n.pt',
        'person_yolov8n-seg.pt':
        mydir +
        '/../models/adetailer/person_yolov8n-seg.pt',
        'person_yolov8s-seg.pt':
        mydir + '/../models/adetailer/person_yolov8s-seg.pt'}

    with change_torch_load():
        pred = predictor(ad_models[type], image, 0.3, device)

    masks = pred_preprocessing(pred)

    return masks


def inpaint(image, mask, pipeline, prompt, n_prompt, inpaint_strength):
    mask_image = mask

    mask_blur = 4
    np_mask = np.array(mask_image)
    kernel_size = 2 * int(4 * 4 + 0.5) + 1
    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur)
    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur)
    mask_image = Image.fromarray(np_mask)

    inpaint_full_res_padding = 32
    mask_image = mask_image.convert('L')
    crop_region = masking.get_crop_region(
        np.array(mask_image), inpaint_full_res_padding)
    crop_region = masking.expand_crop_region(
        crop_region, 512, 512, mask_image.width, mask_image.height)

    init_image = image.crop(crop_region)
    mask_image = mask_image.crop(crop_region)

    original_w = init_image.width
    original_h = init_image.height

    init_image = images.resize_image(2, init_image, 512, 512)
    mask_image = images.resize_image(2, mask_image, 512, 512)

    # init_image.show()
    # mask_image.show()
    result = pipeline(prompt=prompt, width=512, height=512, negative_prompt=n_prompt, image=init_image,
                        strength=inpaint_strength, mask_image=mask_image,  num_inference_steps=28, callback_steps = 1, callback=inpaint_progress).images[0] #, 
    # result.show()
    result = images.resize_image(1, result, original_w, original_h)

    image.paste(result, crop_region)
    # return image
    return image


'''

def start_inpaint_character_pipeline(controlnet, image, device, prompt, n_prompt):
    masks2 = get_inpaint_masks(
        image, "person_yolov8n-seg.pt", 'cuda' if device.type == 'cuda' else 'cpu')

    mask_image = masks2[0]

    mask_blur = 4
    np_mask = np.array(mask_image)
    kernel_size = 2 * int(4 * 4 + 0.5) + 1
    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur)
    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur)
    mask_image = Image.fromarray(np_mask)

    inpaint_full_res_padding = 32
    mask_image = mask_image.convert('L')
    crop_region = masking.get_crop_region(
        np.array(mask_image), inpaint_full_res_padding)
    crop_region = masking.expand_crop_region(
        crop_region, 840, 560, mask_image.width, mask_image.height)

    init_image = image.crop(crop_region)
    mask_image = mask_image.crop(crop_region)

    original_w = init_image.width
    original_h = init_image.height

    init_image = images.resize_image(2, init_image, 840, 560)
    mask_image = images.resize_image(2, mask_image, 840, 560)

    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(
        "../models/deliberate_v3.safetensors",
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
        local_files_only=True,
        clip_skip=2,
        controlnet=controlnet
    )

    pipeline.to('cuda' if device.type == 'cuda' else 'mps')
    setup_pipeline(pipeline, device)
    prompt = "a white woman"
    control_img = openpose(init_image)

    init_image.show()
    mask_image.show()
    control_img.show()

    result = pipeline(prompt=prompt, width=840, height=560, control_image=control_img, negative_prompt=n_prompt,
                      image=init_image, strength=0.4, mask_image=mask_image,  num_inference_steps=28, callback=progress).images[0]
    result = images.resize_image(1, result, original_w, original_h)

    image.paste(result, crop_region)

    return image

'''


def start_inpaint_pipeline(images, batch_count, device, prompt, n_prompt, model_id, lora_id, clip_skip, vae, inpaint_strength):
    # inpaint pipe

    #stablediffusionapi/rev-animated
    if model_id == "deliberate_v2":
        #print("deliberate_v2")
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        #get_inpaint_model_path(model_id),
        "5w4n/deliberate-v2-inpainting",
        #use_safetensors=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        safety_checker=None,
        #load_safety_checker=False,
        local_files_only=True,
        clip_skip=clip_skip
    )
    else:
        #print("none deliberate_v2")
        pipeline = StableDiffusionInpaintPipeline.from_single_file(
        get_inpaint_model_path(model_id),
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        load_safety_checker=False,
        local_files_only=True,
        clip_skip=clip_skip
    )

    pipeline.to('cuda' if device.type == 'cuda' else 'mps')
    if (isXLModel(model_id) == False):
        setup_pipeline_lora(pipeline, lora_id)
        setup_pipeline_vae(pipeline, device, vae)
        setup_pipeline_negtive_embeds(pipeline, device, model_id)

    # masks2 = get_inpaint_masks(image, "person_yolov8n-seg.pt", 'cuda' if device.type == 'cuda' else 'cpu')
    # print(masks2)
    # masks2 = [masks2[0]]
    # image = inpaint_all(image, masks2, pipeline, "a chic Caucasian kid", n_prompt)
    # image.save("test_inpaint_person.png")
    n = batch_count
    for i in range(0, n):
        print("inpaint_start:" + str(i), flush=True)
        masks = get_inpaint_masks(images[i], "face_yolov8n.pt", device)
        j = 0
        for mask in masks:
            print("inpaint_mask_start:" + str(j) + ":"+  str(len(masks)), flush=True)
            j = j + 1
            images[i] = inpaint(images[i], mask, pipeline, prompt, n_prompt, inpaint_strength)

    # image = inpaint_it(pipeline, image, "hand_yolov8n.pt", device)
    return images

'''
def start_controlnet_pipeline(image, depthImage, batch_count, device, prompt, n_prompt, control_net_model, model_id, scheduler_type, lora_id, cfg, clip_skip, sampler_steps, vae, resolution=1024):
    
    model = Model(task_name = control_net_model, device=device,
                  base_model_id=get_model_path(model_id),
                  scheduler_type = scheduler_type,
                  clip_skip=clip_skip,
                  from_pretrained=get_model_path_from_pretrained(model_id),
                  use_xl=isXLModel(model_id))

    if (isXLModel(model_id) == False):
        setup_pipeline_lora(model.pipe, lora_id)
        setup_pipeline_vae(model.pipe, device, vae)
        setup_pipeline_negtive_embeds(model.pipe, device, model_id)

    # model.set_base_model('SdValar/deliberate2')
    # model.set_base_model('stablediffusionapi/deliberate-v2')
    # demo = create_demo(model.process_depth)

    imageresults = []
    n = batch_count
    #print('start_controlnet_pipeline'+ str(n))
    for i in range(0, n):
        print("controlnet_start:" + str(i), flush=True)
        result = model.process_depth(image, image, prompt=prompt, num_images=1, additional_prompt=None, negative_prompt=n_prompt, image_resolution=resolution, preprocess_resolution=resolution,
                                     num_steps=sampler_steps, guidance_scale=cfg, seed=randomize_seed_fn(seed=0, randomize_seed=True), preprocessor_name='Midas', callback=controlnet_progress)
        result[0].save("../../output/temp_depth.png")
        imageresults= [result[1]] + imageresults
        #print("imageresults" + str(len(imageresults)))

    return imageresults
'''

def start_controlnet_pipeline(image, depthImage, batch_count, device, prompt, n_prompt, control_net_model, model_id, scheduler_type, lora_id, cfg, clip_skip, sampler_steps, vae, resolution=1024):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

    use_xl = isXLModel(model_id)
    from_pretrained = get_model_path_from_pretrained(model_id)

    #controlnet_model
    if use_xl:
        print("not ready yet")
    else:
        depth_model_id = CONTROLNET_MODEL_IDS["depth"]
        print('depth_model_id: ' + depth_model_id)
        depth_controlnet = ControlNetModel.from_pretrained(depth_model_id,
                                                            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                                                            #local_files_only=Tru
                                                            
                                                            )
    
        openpose_model_id = CONTROLNET_MODEL_IDS["Openpose"]
        print('openpose_model_id: '+openpose_model_id)
        openpose_controlnet = ControlNetModel.from_pretrained(openpose_model_id,
                                                              
                                                            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                                                            device_map="auto",
                                                            #local_files_only=True
                                                            )
        resolution = 512
        pose_preprocessor = Preprocessor()
        pose_preprocessor.load("Openpose")
        pose_control_image = pose_preprocessor(
                    image=image,
                    image_resolution=resolution,
                    detect_resolution=resolution,
                    hand_and_face=True,
                )

        depth_preprocessor = Preprocessor()
        depth_preprocessor.load("Midas")
        depth_control_image = depth_preprocessor(
                    image=image,
                    image_resolution=resolution,
                    detect_resolution=resolution,
                )

    #pipe
    if use_xl:
        print("not ready yet")
    else:
        if(from_pretrained):
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                get_model_path(model_id),
                safety_checker=None,
                controlnet=[depth_controlnet, openpose_controlnet],
                controlnet_conditioning_scale = [0.5,1.0],
                local_files_only=True,
                clip_skip=clip_skip,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32 
            )
        else:
            pipe = StableDiffusionControlNetPipeline.from_single_file(
                get_model_path(model_id),
                safety_checker = None,
                controlnet = [depth_controlnet, openpose_controlnet],#, openpose_controlnet
                controlnet_conditioning_scale = [0.5,1.0],#, 1.0
                local_files_only=True,
                clip_skip=clip_skip,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32 
            )     
        setup_pipeline_lora(pipe, lora_id)
        setup_pipeline_vae(pipe, device, vae)
        setup_pipeline_negtive_embeds(pipe, device, model_id)
        
    
    #scheduler
    if(scheduler_type == "DPM++2MK"):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    elif(scheduler_type == "DPM++2SK"):
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    elif(scheduler_type == "DPM++SDEK"):
        pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    #settings
    pipe.to(device)
    torch.cuda.empty_cache()
    gc.collect()
    #start pipe

    imageresults = []
    for i in range(0, batch_count):
        seed=randomize_seed_fn(seed=0, randomize_seed=True)
        generator = torch.Generator().manual_seed(seed)
        results = pipe(prompt=prompt,
                negative_prompt=n_prompt,
                num_inference_steps=sampler_steps,
                generator=generator,
                guidance_scale=cfg,
                callback=controlnet_progress,
                callback_steps = 1,
                image=[depth_control_image, pose_control_image]).images #, pose_control_image
        imageresults= [results[0]] + imageresults
    return imageresults


def setup_pipeline_negtive_embeds(pipe, device, model_id):
     # negative embedding
    
    pipe.load_textual_inversion("../models/negative_embeddings/bad_prompt_version2.pt",
                                token="bad_prompt_version2",
                                local_files_only=True,)
    # model.pipe.load_textual_inversion("models/negative_embeddings/bad-artist-anime.pt", token="bad-artist-anime",
    #                                local_files_only=True,)
    pipe.load_textual_inversion("../models/negative_embeddings/bad-artist.pt",
                                token="bad-artist",
                                local_files_only=True,)
    pipe.load_textual_inversion("../models/negative_embeddings/bad-hands-5.pt",
                                token="bad-hands-5",
                                local_files_only=True,)
    pipe.load_textual_inversion("../models/negative_embeddings/ng_deepnegative_v1_75t.pt",
                                token="ng_deepnegative_v1_75t",
                                local_files_only=True,)
    pipe.load_textual_inversion("../models/negative_embeddings/easynegative.safetensors",
                                token="easynegative",
                                local_files_only=True,)
    
    if(model_id == "realisticVision"):
         print("load UnrealisticDream")
         pipe.load_textual_inversion("../models/negative_embeddings/UnrealisticDream.pt",
                                token="UnrealisticDream",
                                local_files_only=True,)
         
    if(model_id == "Arthemy Comics"):
         print("load Arthemy Comics")
         pipe.load_textual_inversion("../models/negative_embeddings/verybadimagenegative_v1.3.pt",
                                token="verybadimagenegative_v1.3",
                                local_files_only=True,)


def setup_pipeline_lora(pipe, lora_id):
    # lora
    if (lora_id != "None"):
        pipe.load_lora_weights("../models/lora", weight_name=get_lora(lora_id),local_files_only=True, lora_scale = 0.7)

def setup_pipeline_vae(pipe, device, vae):
    # vae
    if(vae!= "None"):
        vae = AutoencoderKL.from_single_file("../models/vae/vae-ft-mse-840000-ema-pruned.safetensors",
                                         local_files_only=True,
                                         use_safetensors=True,
                                         torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
        vae.to('cuda' if device.type == 'cuda' else 'mps')
        pipe.vae = vae

   

def get_model_path_from_pretrained(model_id):
    if( model_id == "revAnimated"):
        return True
    elif(model_id == "realitycheckXL"):
        return False
    return False

def isXLModel(model_id):
    if(model_id == "realitycheckXL"):
        return True
    if(model_id == "sd_xl_base"):
        return True
    return False

def get_model_path(model_id):
    if (model_id == "realisticVision"):
        return "../models/realisticVisionV51_v51VAE.safetensors"
    elif (model_id == "revAnimated"):
        return "stablediffusionapi/rev-animated"
    elif (model_id == "Arthemy Comics"):
        return "../models/arthemycomics.safetensors"
    elif (model_id == "sd_xl_base"):
        return "../models/sd_xl_base_1.0.safetensors"
    else:
        return "../models/"+model_id+".safetensors"


def get_inpaint_model_path(model_id):
    if (model_id == "realisticVision"):
        return "../models/realisticVisionV51_v51VAE-inpainting.safetensors"
    elif (model_id == "revAnimated"):
        return "../models/revAnimated_v121Inp-inpainting.safetensors"
    elif (model_id == "deliberate_v2"):
        return "../models/deliberate_v3-inpainting.safetensors"
    elif (model_id == "Arthemy Comics"):
        return "../models/arthemycomics-inpainting.safetensors"
    elif (model_id == "realitycheckXL"):
        return "../modes/realitycheckXL.safetensors"
    else:
        return "../models/"+model_id+"-inpainting.safetensors"


def get_lora(lora_id):
    if (lora_id == "empty"):
        return None
    elif (lora_id == "Jim Lee"):
        return "jim_lee_offset_right_filesize.safetensors"
    else:
        return lora_id+".safetensors"
    

def get_styled_prompt(style, prompt):
    print("get_styled_prompt" + style)
    style = "{prompt}"
    if(style == "painterly"):
         #style = "{prompt}"
        style = "Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic"
    elif(style == "pencil"):
        style  = "{prompt}, Pencil sketch drawing . black and white painting, fantasy art, photo realistic, dynamic lighting, artstation, poster, volumetric lighting, very detailed, 4k, da vinci style made with a charcoal pencil"
        #style = "line art drawing {prompt} . professional, sleek, modern, minimalist, graphic, line art, vector graphics"
    elif(style=="cinematic"):
        style = "cinematic film still {prompt} . shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy"
    elif(style=="photoreal"):
        style = "cinematic photo {prompt} . 35mm photograph, film, bokeh, professional, 4k, highly detailed"

    return style.replace("{prompt}", prompt)

def get_styled_neg_prompt(style, prompt):
    style = ""
    if(style == "painterly"):
        style = ""
        #style = "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    elif(style == "pencil"):
        style = "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
    elif(style=="cinematic"):
        style = "anime, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"
    elif(style=="photoreal"):
        style = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly"
    return style + prompt 

def get_prompt(model_id):
    #style =  "Watercolor painting {prompt} . Vibrant, beautiful, painterly, detailed, textural, artistic"
    prompt = ""
    if(model_id == "realisticVision"):
        prompt = "RAW photo, subject, 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT3"
    elif (model_id == "Arthemy Comics"):
        prompt = "(artwork:1.2),(lineart:1.33),[CHARACTER | SETTING | EMOTIONS],(hyperdefined),(inked-art),[COLORS | AESTHETIC],complex lighting,(flat colors),ultradetailed,(fine-details:1.2),absurdres,(atmosphere)"
     
    #prompt = style.replace(" {prompt}", prompt)
    return prompt

def get_neg_prompt(model_id):
    #style = "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
    prompt = ""
    if(model_id == "realisticVision"):
        prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream" #
    elif (model_id == "Arthemy Comics"):
        prompt = "bad-hands-5, verybadimagenegative_v1.3, (sketch),lowres,(text,words:1.3),watermark,(simple background),glitch,(jpeg-artifact:1.2),compressed, jpeg" #
    prompt = "Blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, Low quality, Bad quality, Long neck, bad_prompt_version2, bad-artist, bad-hands-5, ng_deepnegative_v1_75t, easynegative"

    #prompt = style + prompt
    return prompt

def get_lora_prompt(lora_id, promot):
    lora_prompt = ""
    if(lora_id == "Drawing"):
        lora_prompt = "drawing, style by NTY"
    
    return lora_prompt + promot



def main(image_id, use_inpaint, use_depth_map, batch_count, prompt, control_net_model, model_id, scheduler_type, lora_id, cfg, clip_skip, sampler_steps, vae, inpaint_strength, use_style, style):
   
    # prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
   
    if use_style:
        lora_id = "None"
        vae="None"
        if(style == "painterly"):
            model_id = "deliberate_v2"
            lora_id = "Painterly-Style_LoRA"
            scheduler_type = "DPM++2MK"
            sampler_steps = 25
            cfg = 7
            clip_skip = 2
        elif(style == "pencil"):
            model_id = "deliberate_v2"
            lora_id = "Drawing"
            scheduler_type = "DPM++2MK"
            sampler_steps = 25
            cfg = 6
        elif(style=="cinematic"):
            model_id = "realisticVision"
            lora_id = "CineStyle5"
        elif(style=="photoreal"):
            model_id = "realisticVision"
        else:
            model_id = "deliberate_v2"
    
    prompt = prompt + get_prompt(model_id)
    prompt = get_styled_prompt(style, prompt)
    prompt = get_lora_prompt(lora_id, prompt)

    n_prompt = get_neg_prompt(model_id)
    n_prompt = get_styled_neg_prompt(style, n_prompt)

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    image = Image.open("../../capture/" + image_id + ".png")
    
    if use_depth_map:
        depth_image = Image.open("../../capture/" + image_id + "_grid.png")
    else:
        depth_image = image


    resolution = int(512.0 / image.height  * image.width)
    resolution = min(MAX_IMAGE_RESOLUTION, resolution)

    results = start_controlnet_pipeline(
        image, depth_image, batch_count, device, prompt, n_prompt, control_net_model, model_id, scheduler_type, lora_id, cfg, clip_skip, sampler_steps, vae, resolution)

    # controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",
    #                                                 torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    #                                                 local_files_only=True)

    torch.cuda.empty_cache()
    gc.collect()
    images = results
    # image.save("test_before.png")
    # image = start_inpaint_character_pipeline(controlnet, image, device, prompt, n_prompt)
    # image.save("test_inpaint.png")
    if(use_inpaint):
        images = start_inpaint_pipeline(
            images, batch_count, device, prompt, n_prompt, model_id, lora_id, clip_skip, vae, inpaint_strength)
        print(f"time -2: {time.time() - start_time}")

    torch.cuda.empty_cache()
    gc.collect()
    
    n = batch_count
    for i in range(0, n):
        meta = PngInfo()
        meta.add_text("prompt", prompt)
        meta.add_text("nprompt", n_prompt)
        meta.add_text("model", get_model_path(model_id))
        meta.add_text("in paintmodel", get_inpaint_model_path(model_id))
        meta.add_text("lora", get_lora(lora_id))
        meta.add_text("cfg", str(cfg))
        meta.add_text("clip skip", str(clip_skip))
        meta.add_text("sampler steps", str(sampler_steps))
        meta.add_text("vae", str(inpaint_strength))
        meta.add_text("control net model", str(control_net_model))
        meta.add_text("scheduler type", str(scheduler_type))
        print("image_save: " + str(i))
        images[i].save("../../output/" + image_id + "_"+ str(i) +
                       ".png", format="PNG", pnginfo=meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node', '-n', type=int, default=0,
                        help="if executes from node")
    parser.add_argument('--image', '-i', type=str, help="image name")
    parser.add_argument('--prompt', '-p', type=str, help="prompt")
    parser.add_argument('--model', '-m', type=str, help="model id")
    parser.add_argument('--cfg', '-c', type=int, help="CFG scale")
    parser.add_argument('--clipskip', '-cs', type=int, help="clip skip")
    parser.add_argument('--sampler_step', '-ss',
                        type=int, help="sampler steps")
    parser.add_argument('--lora', '-l', type=str, help="lora id")
    parser.add_argument('--batch_count', '-b', type=int, help = "batch count", default = 1)

    parser.add_argument('--vae','-v', type=int, default=0, help="if use vae")  # default=False, h
    parser.add_argument('--control_net_model','-cnm', type=str, default="depth", help="control net model")
    parser.add_argument('--scheduler','-s', type=str, help="scheduler")
    parser.add_argument('--inpaint_strength','-is', type=float, default=0.4, help="inpaint strength")
    parser.add_argument('--use_depth_map','-d', type=int, default=0, help="if use depth map")
    parser.add_argument('--use_inpaint', '-ip', type=int, default=1, help="if use inpaint")
    
    #style, painterly, pencil, cinematic, photoreal
    parser.add_argument('--use_style', '-us', type=int, default = 0, help="if use style")
    parser.add_argument('--style', '-st', type=str, default = "painterly", help="the style")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('node: ' + str(args.node))
    print('arg_image_id: ' + args.image)
    print('arg_prompt: ' + args.prompt)
    print('arg_model_id: ' + args.model)
    print('cfg: ' + str(args.cfg))
    print('clip_skip: ' + str(args.clipskip))
    print('sampler_steps: ' + str(args.sampler_step))
    print('lora_id: ' + args.lora)
    print('batch_count:  '+ str(args.batch_count))

    print ('vae: ' + str(args.vae > 0))
    print ('control_net_model: ' + str(args.control_net_model))
    # canny, depth, scribble
    print ('scheduler: ' + str(args.scheduler))
    # DPM++2MK, DPM++2SK, DPM++SDEK, EULARA
    print ('inpaint_strength: ' + str(args.inpaint_strength))

    print ('use_depth_magp: ' + str(args.use_depth_map > 0))
    print ('use_inpaint: ' + str(args.use_inpaint > 0))

    ##
    print ('use_style' + str (args.use_style > 0))
    print ('style' + args.style)

   

    #eular
    #DPM++ 2M Karras
    #DPM++ SDE Karras
    if (args.node == 1):
        mydir = os.getcwd()
        mydir_tmp = mydir + "/../scripts/cli"
        mydir_new = os.chdir(mydir_tmp)

    main(args.image, args.use_inpaint > 0, args.use_depth_map >0, args.batch_count, args.prompt, args.control_net_model, args.model, args.scheduler, args.lora,
         args.cfg, args.clipskip, args.sampler_step, args.vae > 0, args.inpaint_strength, args.use_style, args.style)
