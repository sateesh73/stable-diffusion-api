import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

def getModel(model_id, device, access_token):
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = DiffusionPipeline.from_pretrained(model_id, safety_checker=None, token=access_token, scheduler=scheduler, torch_dtype=torch.float16)
    
    pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    return pipe

def genrate_image(prompt, height, width, n_prompt, scale, num_images_per_prompt, steps, pipe, device):
    seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(prompt, num_inference_steps=steps, height=height, width=width,guidance_scale = scale, num_images_per_prompt=num_images_per_prompt, negative_prompt=n_prompt, generator=generator).images[0]
    return image