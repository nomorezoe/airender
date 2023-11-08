

pipe = diffusers.StableDiffusionPipeline.from_single_file("sample.safetensors")
pipe.save_pretrained("output folder name")