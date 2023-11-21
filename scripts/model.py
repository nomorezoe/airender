from __future__ import annotations

import gc

import numpy as np
import PIL.Image
import torch
from controlnet_aux.util import HWC3
from diffusers import (ControlNetModel, DiffusionPipeline,
                       StableDiffusionControlNetPipeline,
                       DPMSolverMultistepScheduler,
                       DPMSolverSinglestepScheduler,
                       EulerAncestralDiscreteScheduler,
                       StableDiffusionXLControlNetPipeline,
                       DPMSolverSDEScheduler)

from cv_utils import resize_image
from preprocessor import Preprocessor
from settings import MAX_IMAGE_RESOLUTION, MAX_NUM_IMAGES

CONTROLNET_MODEL_IDS = {
    'Openpose': 'lllyasviel/control_v11p_sd15_openpose',
    'canny': 'lllyasviel/control_v11p_sd15_canny',
    'MLSD': 'lllyasviel/control_v11p_sd15_mlsd',
    'scribble': 'lllyasviel/control_v11p_sd15_scribble',
    'softedge': 'lllyasviel/control_v11p_sd15_softedge',
    'segmentation': 'lllyasviel/control_v11p_sd15_seg',
    'depth': 'lllyasviel/sd-controlnet-depth',#control_v11f1p_sd15_depth',
    'NormalBae': 'lllyasviel/control_v11p_sd15_normalbae',
    'lineart': 'lllyasviel/control_v11p_sd15_lineart',
    'lineart_anime': 'lllyasviel/control_v11p_sd15s2_lineart_anime',
    'shuffle': 'lllyasviel/control_v11e_sd15_shuffle',
    'ip2p': 'lllyasviel/control_v11e_sd15_ip2p',
    'inpaint': 'lllyasviel/control_v11e_sd15_inpaint',
}
CONTROLNET_MODEL_XL_IDS = {
    'depth':'diffusers/controlnet-depth-sdxl-1.0',
}

def download_all_controlnet_weights() -> None:
    ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
    ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_scribble")
    #for model_id in CONTROLNET_MODEL_IDS.values():
    #   ControlNetModel.from_pretrained(model_id)


class Model:
    def __init__(self,
                 base_model_id: str = 'models/deliberate_v3.safetensors',
                 task_name: str = 'Canny',
                 device: str = 'cuda',
                 scheduler_type: str = "DPM2M++K",
                 clip_skip: int = 1,
                 from_pretrained: bool = False,
                 use_xl: bool = False):
        self.device = device
        self.base_model_id = ''
        self.task_name = ''
        self.scheduler_type = scheduler_type
        self.clip_skip = clip_skip
        self.from_pretrained = from_pretrained
        self.use_xl = use_xl
        self.pipe = self.load_pipe(base_model_id, task_name, scheduler_type)
        self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str, task_name, scheduler_type) -> DiffusionPipeline:
        if base_model_id == self.base_model_id and task_name == self.task_name and hasattr(
                self, 'pipe') and self.pipe is not None:
            return self.pipe
        if self.use_xl:
            model_id = CONTROLNET_MODEL_XL_IDS[task_name]
        else:
            model_id = CONTROLNET_MODEL_IDS[task_name]
        print("model id " + model_id)
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                                                     local_files_only=True)
        if self.from_pretrained:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            safety_checker=None,
            controlnet=controlnet,
            local_files_only=True,
            clip_skip=self.clip_skip,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32 
            )

        else:    
            print("use_xl" + str(self.use_xl))
            print("base_model_id" + base_model_id)
            if self.use_xl:
                pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                    base_model_id,
                    use_safetensors=True, 
                    load_safety_checker=False,
                    controlnet=controlnet,
                    local_files_only=True,
                    clip_skip = self.clip_skip,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
            
            else:
                pipe = StableDiffusionControlNetPipeline.from_single_file(
                    base_model_id,
                    use_safetensors=True, 
                    load_safety_checker=False,
                    controlnet=controlnet,
                    local_files_only=True,
                    clip_skip = self.clip_skip,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
        #sampler
        if(scheduler_type == "DPM++2MK"):
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        elif(scheduler_type == "DPM++2SK"):
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        elif(scheduler_type == "DPM++SDEK"):
            pipe.scheduler = DPMSolverSDEScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="dpmsolver++")
        else:
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


        #if self.device.type == 'cuda':
            #torch.autocast("cuda")
            #pipe.enable_xformers_memory_efficient_attention()
        pipe.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.base_model_id = base_model_id
        self.task_name = task_name
        return pipe

    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return
        if self.pipe is not None and hasattr(self.pipe, 'controlnet'):
            del self.pipe.controlnet
        torch.cuda.empty_cache()
        gc.collect()
        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id,
                                                     torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                                                     local_files_only=True)
        controlnet.to(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        self.pipe.controlnet = controlnet
        self.task_name = task_name

    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        if not prompt:
            prompt = additional_prompt
        else:
            prompt = f'{prompt}, {additional_prompt}'
        return prompt

    @torch.autocast('cuda')
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    ) -> list[PIL.Image.Image]:
        generator = torch.Generator().manual_seed(seed)
        return self.pipe(prompt=prompt,
                         negative_prompt=negative_prompt,
                         guidance_scale=guidance_scale,
                         num_images_per_prompt=num_images,
                         num_inference_steps=num_steps,
                         generator=generator,
                         callback=callback,
                         image=control_image).images

    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        low_threshold: int,
        high_threshold: int,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load('Canny')
        control_image = self.preprocessor(image=image,
                                          low_threshold=low_threshold,
                                          high_threshold=high_threshold,
                                          detect_resolution=image_resolution)

        self.load_controlnet_weight('Canny')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_mlsd(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load('MLSD')
        control_image = self.preprocessor(
            image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
            thr_v=value_threshold,
            thr_d=distance_threshold,
        )
        self.load_controlnet_weight('MLSD')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_scribble(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name == 'HED':
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=False,
            )
        elif preprocessor_name == 'PidiNet':
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=False,
            )
        self.load_controlnet_weight('scribble')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_scribble_interactive(
        self,
        image_and_mask: dict[str, np.ndarray],
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if image_and_mask is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        image = image_and_mask['mask']
        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)

        self.load_controlnet_weight('scribble')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_softedge(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ['HED', 'HED safe']:
            safe = 'safe' in preprocessor_name
            self.preprocessor.load('HED')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            )
        elif preprocessor_name in ['PidiNet', 'PidiNet safe']:
            safe = 'safe' in preprocessor_name
            self.preprocessor.load('PidiNet')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=safe,
            )
        else:
            raise ValueError
        self.load_controlnet_weight('softedge')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_openpose(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load('Openpose')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                hand_and_face=True,
            )
        self.load_controlnet_weight('Openpose')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_segmentation(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight('segmentation')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_depth(
        self,
        image: np.ndarray,
        depthImage: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            #image = np.array(image)
            image = HWC3(depthImage)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=depthImage,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight('depth')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_normal(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load('NormalBae')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        self.load_controlnet_weight('NormalBae')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_lineart(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name in ['None', 'None (anime)']:
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ['Lineart', 'Lineart coarse']:
            coarse = 'coarse' in preprocessor_name
            self.preprocessor.load('Lineart')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                coarse=coarse,
            )
        elif preprocessor_name == 'Lineart (anime)':
            self.preprocessor.load('LineartAnime')
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )
        if 'anime' in preprocessor_name:
            self.load_controlnet_weight('lineart_anime')
        else:
            self.load_controlnet_weight('lineart')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_shuffle(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == 'None':
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        else:
            self.preprocessor.load(preprocessor_name)
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
            )
        self.load_controlnet_weight('shuffle')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results

    @torch.inference_mode()
    def process_ip2p(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)
        self.load_controlnet_weight('ip2p')
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        return [control_image] + results
