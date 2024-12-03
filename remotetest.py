# Importing necessary libraries for image processing, base64 encoding, 
# machine learning pipelines, and type handling
import base64
import io
from typing import Optional
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler, StableDiffusionXLImg2ImgPipeline, PNDMScheduler
from pydantic import BaseModel
import torch
from PIL import Image
from io import BytesIO
import tempfile
from accelerate import Accelerator
from accelerate.logging import get_logger

# Create a temporary directory for caching model files
temp_dir = tempfile.TemporaryDirectory()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a Pydantic model for input validation and type checking
# This ensures that the input data has the correct structure and types
class Item(BaseModel):
    prompt: str  # Main text prompt for image generation
    prompt_2: Optional[str] = ""  # Secondary prompt (optional)
    base64_image: str  # Input image encoded in base64
    model_id: Optional[str]  # Optional model identifier
    # Various generation parameters with default values
    strength: Optional[int] = 0.75  # Image modification strength
    guidance_scale: Optional[int] = 7.5  # How closely to follow the prompt
    num_inference_steps: Optional[int] = 15  # Number of denoising steps
    guidance_rescale: Optional[float] = 0.7
    aesthetic_score: Optional[float] = 7.0
    negative_aesthetic_score: Optional[float] = 2.5
    # Negative prompts to guide what NOT to include in the image
    negative_prompt: Optional[str] = "ugly, pixelated, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, low quality, poor lighting, dull, and unclear"
    negative_prompt_2: Optional[str] = "low quality, different face, cartoonish, distorted anatomy"

# Initial model configuration
init_model_id = "stablediffusionapi/deliberate-v2"

# Load the Stable Diffusion XL Refiner model with specific optimizations
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0", 
    torch_dtype=torch.float16,  # Use half-precision for memory efficiency
    variant="fp16", 
    use_safetensors=True,
    cache_dir=temp_dir.name,  # Use temporary directory for caching
)

# Configure and optimize the model
pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()  # Optimize memory usage
pipe.enable_model_cpu_offload()  # Move model to CPU when not in use
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)  # Compile for performance
pipe = pipe.to("cuda")  # Move to CUDA for GPU acceleration

# Initialize Accelerator for distributed computing and logging
accelerator = Accelerator()
logger = get_logger(__name__)

def predict(item, run_id, logger, binaries=None):
    """
    Main prediction function for image generation/modification
    
    Args:
    - item: Input parameters for image generation
    - run_id: Unique run identifier
    - logger: Logging object
    - binaries: Optional additional binary data
    
    Returns:
    - List of generated images encoded in base64
    """
    logger.info("remote call received")
    
    # Validate and parse input using Pydantic model
    item = Item(**item)
    model_id = item.model_id
    
    # Decode and prepare input image
    init_image = Image.open(BytesIO(base64.b64decode(item.base64_image))).convert("RGB")
    init_image = init_image.resize((768, 512))  # Resize image to standard dimensions

    # Optionally load a different model if model_id is provided
    if model_id:
        logger.info("reload new model")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            cache_dir=temp_dir.name
        ).to(device)
        
        # Configure the scheduler for the new model
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Apply additional optimizations
        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_attention_slicing()

    try:
        # Generate images using the pipeline
        images = pipe(
            prompt=item.prompt, 
            prompt_2=item.prompt_2, 
            image=init_image, 
            strength=item.strength, 
            guidance_scale=item.guidance_scale,
            negative_prompt=item.negative_prompt,
            negative_prompt_2=item.negative_prompt_2,
            num_inference_steps=item.num_inference_steps
        ).images
    except TypeError as e:
        # Log any errors during image generation
        logger.error(f"Error in pipeline call: {e}{item}")

    # Process and encode generated images
    if images is not None:
        logger.info("finalizing return images")
        finished_images = []
    
        for image in images:
            # Convert each image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            finished_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))

        return finished_images
    else:
        return {"result": False}