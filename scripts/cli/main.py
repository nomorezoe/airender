# ,ControlNetModel，StableDiffusionControlNetInpaintPipeline
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


# from controlnet_aux import OpenposeDetector
import argparse
import images
import cv2
import numpy as np
import time
import torch
import masking
from adetailer import ultralytics_predict
from adetailer.mask import filter_by_ratio, mask_preprocess, sort_bboxes
from adetailer.common import PredictOutput
from utils import randomize_seed_fn
from model import Model
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from contextlib import contextmanager




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
        torch.load = torch.load  # safe.unsafe_torch_load
        yield
    finally:
        torch.load = orig


def get_inpaint_masks(image, type, device):
    predictor = ultralytics_predict
    mydir = os.getcwd()
    print(mydir)
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


def inpaint_all(image, masks, pipeline, prompt, n_prompt):

    for mask in masks:
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
            crop_region, 840, 560, mask_image.width, mask_image.height)

        init_image = image.crop(crop_region)
        mask_image = mask_image.crop(crop_region)

        original_w = init_image.width
        original_h = init_image.height

        init_image = images.resize_image(2, init_image, 840, 560)
        mask_image = images.resize_image(2, mask_image, 840, 560)

        # init_image.show()
        # mask_image.show()
        result = pipeline(prompt=prompt, width=840, height=560, negative_prompt=n_prompt, image=init_image,
                          strength=0.4, mask_image=mask_image,  num_inference_steps=28, callback=progress).images[0]
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


def start_inpaint_pipeline(image, device, prompt, n_prompt, model_id, lora_id, clip_skip):
    # inpaint pipo
    pipeline = StableDiffusionInpaintPipeline.from_single_file(
        get_inpaint_model_path(model_id),
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        load_safety_checker=False,
        local_files_only=True,
        clip_skip=clip_skip
    )

    pipeline.to('cuda' if device.type == 'cuda' else 'mps')
    setup_pipeline(pipeline, device, lora_id)

    # masks2 = get_inpaint_masks(image, "person_yolov8n-seg.pt", 'cuda' if device.type == 'cuda' else 'cpu')
    # print(masks2)
    # masks2 = [masks2[0]]
    # image = inpaint_all(image, masks2, pipeline, "a chic Caucasian kid", n_prompt)
    # image.save("test_inpaint_person.png")

    masks = get_inpaint_masks(image, "face_yolov8n.pt", device)
    image = inpaint_all(image, masks, pipeline, prompt, n_prompt)

    # image = inpaint_it(pipeline, image, "hand_yolov8n.pt", device)
    return image


def start_controlnet_pipeline(image, device, prompt, n_prompt, model_id, lora_id, cfg, clip_skip, sampler_steps):
    model = Model(task_name='depth', device=device,
                  base_model_id=get_model_path(model_id),
                  clip_skip=clip_skip)

    setup_pipeline(model.pipe, device, lora_id)

    # model.set_base_model('SdValar/deliberate2')
    # model.set_base_model('stablediffusionapi/deliberate-v2')
    # demo = create_demo(model.process_depth)

    image_results = model.process_depth(image, prompt=prompt, num_images=1, additional_prompt=None, negative_prompt=n_prompt, image_resolution=840, preprocess_resolution=512,
                                        num_steps=sampler_steps, guidance_scale=cfg, seed=randomize_seed_fn(seed=0, randomize_seed=True), preprocessor_name='Midas', callback=progress)

    return image_results[1]


def setup_pipeline(pipe, device, lora_id):
    # lora
    if (lora_id != "None"):
        pipe.load_lora_weights("../models/lora", weight_name=get_lora(lora_id))

    # vae
    vae = AutoencoderKL.from_single_file("../models/vae/vae-ft-mse-840000-ema-pruned.safetensors",
                                         local_files_only=True,
                                         use_safetensors=True,
                                         torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    vae.to('cuda' if device.type == 'cuda' else 'mps')
    #pipe.vae = vae

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


def get_model_path(model_id):
    return "../models/"+model_id+".safetensors"


def get_inpaint_model_path(model_id):
    if (model_id == "deliberate_v2"):
        return "../models/deliberate_v3-inpainting.safetensors"
    return "../models/"+model_id+"-inpainting.safetensors"


def get_lora(lora_id):
    if (lora_id == "empty"):
        return None
    return lora_id+".safetensors"


def main(image_id, prompt, model_id, lora_id, cfg, clip_skip, sampler_steps):

    # prompt = "20-year-old African American woman and a chic Caucasian woman, in New York park, reminiscent of a Nike commercial. Warm, golden hues envelop the scene, highlighting their determined expressions. The soft, natural light adds a cinematic touch to the atmosphere, Photography, inspired by Gordon Parks."
    n_prompt = "bad_prompt_version2, bad-artist, bad-hands-5, ng_deepnegative_v1_75t, easynegative"

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    image = Image.open("../../capture/" + image_id + ".png")

    image = start_controlnet_pipeline(
        image, device, prompt, n_prompt, model_id, lora_id, cfg, clip_skip, sampler_steps)

    # controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose",
    #                                                 torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
    #                                                 local_files_only=True)

    # image.save("test_before.png")
    # image = start_inpaint_character_pipeline(controlnet, image, device, prompt, n_prompt)
    # image.save("test_inpaint.png")

    image = start_inpaint_pipeline(
        image, device, prompt, n_prompt, model_id, lora_id, clip_skip)
    print(f"time -2: {time.time() - start_time}")

    meta = PngInfo()
    meta.add_text("prompt", prompt)
    meta.add_text("nprompt", n_prompt)
    meta.add_text("model", get_model_path(model_id))
    meta.add_text("in paintmodel", get_inpaint_model_path(model_id))
    meta.add_text("lora", get_lora(lora_id))
    meta.add_text("cfg", str(cfg))
    meta.add_text("clip skip", str(clip_skip))
    meta.add_text("sampler steps", str(sampler_steps))

    image.save("../../output/" + image_id +
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print('node: ' + str(args.node))
    print('arg_image_id: ' + args.image)
    print('arg_prompt: ' + args.prompt)
    print('arg_model_id: ' + args.model)
    print('lora_id: ' + args.lora)
    print('cfg: ' +  str(args.cfg))
    print('clip_skip: ' +  str(args.clipskip))
    print('sampler_steps: ' + str(args.sampler_step))

    if (args.node == 1):
        mydir = os.getcwd()
        mydir_tmp = mydir + "/../scripts/cli"
        mydir_new = os.chdir(mydir_tmp)

    main(args.image, args.prompt, args.model, args.lora,
         args.cfg, args.clipskip, args.sampler_step)
