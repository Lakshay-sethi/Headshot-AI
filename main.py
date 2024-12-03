import base64
import io
from typing import Optional
#import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline,UniPCMultistepScheduler,StableDiffusionXLImg2ImgPipeline, PNDMScheduler
from pydantic import BaseModel
import torch
from PIL import Image
from io import BytesIO
import tempfile
from accelerate import Accelerator
from accelerate.logging import get_logger

temp_dir = tempfile.TemporaryDirectory()


class Item(BaseModel):
    prompt: str
    prompt_2: Optional[str] = ""
    base64_image: str
    model_id: Optional[str]
    strength: Optional[int] = 0.75
    guidance_scale: Optional[int] = 7.5
    num_inference_steps: Optional[int] = 15
    guidance_rescale : Optional[float] = 0.7
    aesthetic_score: Optional[float] = 7.0
    negative_aesthetic_score : Optional[float] = 2.5
    negative_prompt: Optional[str] = "ugly, pixalated, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, low quality, poor lighting, dull, and unclear"
    negative_prompt_2 : Optional[str] = "low quality, different face, cartoonish, distored anatomy"

# init model weight

init_model_id = "stablediffusionapi/deliberate-v2"
#pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,cache_dir=temp_dir.name,
)
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe = pipe.to("cuda")



# Initialize Accelerator
accelerator = Accelerator()
logger = get_logger(__name__)

def predict(item, run_id, logger, binaries=None):
    logger.info("remote call recieved")
    global model
    #logger.info(item)
    item = Item(**item) 
    model_id = item.model_id         
    init_image = Image.open(BytesIO(base64.b64decode(item.base64_image))).convert("RGB")
    init_image = init_image.resize((768, 512))

    #custom model weight if need
    if model_id:
        logger.info("reload new model")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir=cache_dir).to(device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)

        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()
        

    try:
        images = pipe(
        prompt=item.prompt, 
        prompt_2=item.prompt_2, 
        image=init_image, 
        strength=item.strength, 
        guidance_scale=item.guidance_scale,
        negative_prompt = item.negative_prompt,
        negative_prompt_2=item.negative_prompt_2,
        num_inference_steps=item.num_inference_steps
        ).images
    except TypeError as e:
        logger.error(f"Error in pipeline call: {e}{item}")

    if images is not None:
            logger.info("finalizing return images")       
            finished_images = []
    
            for image in images:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

            return finished_images
    else:
            return {"result": False}