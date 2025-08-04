# generate_image.py
# This script uses the fal-ai API to generate an image from a user-provided text prompt.
# It supports multiple models: the original CyberRealistic Pony and the new FLUX.1 Kontext.

import argparse
import fal_client
import sys
import os
import requests
import uuid
from PIL import Image
import io

# Load Fal-AI API key from falkey.txt
try:
    with open("falkey.txt", "r") as f:
        os.environ["FAL_KEY"] = f.read().strip()
except FileNotFoundError:
    print("ERROR: falkey.txt not found. Please create this file and place your Fal API key in it.")
    sys.exit(1)


def on_queue_update(update):
    """
    This function is a callback that gets triggered by the fal_client.subscribe method.
    It's used to print real-time logs from the image generation process, which helps in monitoring the progress.
    """
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

# --- Model Option 1: CyberRealistic Pony ---
# This is the original function, renamed for clarity.
def generate_image_with_pony(prompt: str) -> dict:
    """
    This function sends a request to the fal-ai API to generate an image using the CyberRealistic Pony model.
    It takes a text prompt and uses the 'fal-ai/lora' endpoint with a specific SDXL-based model that's designed
    for photorealistic image generation. The prompt is automatically formatted with the special tokens and 
    structure required by Pony Diffusion models.

    Args:
        prompt: The text description of the image to be generated.

    Returns:
        A dictionary containing the API's response, which includes the URL of the generated image.
    """
    print("🎨 Starting image generation with CyberRealistic Pony model...")
    
    # Create the full prompt by inserting the user's prompt into the Pony Diffusion template
    # The "score_9, score_8_up, score_7_up" tokens are special quality tags used by Pony models
    # "BREAK" is used to separate different parts of the prompt for better parsing
    full_prompt = f"score_9, score_8_up, score_7_up, BREAK, ((score_9, score_8_up, score_7_up, epic, gorgeous, film grain, grainy, {prompt}, detailed face, (detailed skin pores:1.1), film grain, low light, detailed eyes, detailed skin, (photorealistic:1.1), (amateur photography:1.1)"
    
    try:
        result = fal_client.subscribe(
            "fal-ai/lora",
            arguments={
                "prompt": full_prompt,
                "clip_skip": 2,
                "scheduler": "DPM++ 2M SDE Karras",
                "image_size": {"width": 1024, "height": 1024},
                "model_name": "John6666/cyberrealistic-pony-v8-sdxl",
                "num_images": 1,
                "image_format": "png",
                "guidance_scale": 5,
                "enable_safety_checker": False,
                "negative_prompt": "score_1, score_2, score_3, text, watermarks, child, youth, underage, teddy bears, stuffed animals, ",
                "prompt_weighting": True,
                "num_inference_steps": 30
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        print("✅ Image generation successful!")
        return result
    except Exception as e:
        print(f"❌ An error occurred during image generation: {e}", file=sys.stderr)
        sys.exit(1)

# --- Model Option 2: FLUX.1 Kontext [Max] ---
# This is the new function for the Kontext model.
def generate_image_with_kontext(prompt: str, image_url: str = None, aspect_ratio: str = "1:1") -> dict:
    """
    This function sends a request to the fal-ai API to generate an image using the FLUX.1 Kontext [Max] model.
    This model can be used for text-to-image generation or for editing an existing image by providing a URL.

    Args:
        prompt: The text description of the image to be generated or the editing instruction.
        image_url: Optional URL of an image to edit.
        aspect_ratio: The desired aspect ratio of the output image.

    Returns:
        A dictionary containing the API's response, which includes the URL of the generated image.
    """
    print("🎨 Starting image generation with FLUX.1 Kontext [Max] model...")
    
    # Prepare arguments for the API call. The prompt here is used directly without special formatting.
    arguments = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "guidance_scale": 3.5,  # Default recommended for FLUX models
        "num_images": 1,
        "output_format": "png",
        "enable_safety_checker": False, # For consistency with the pony implementation
    }

    # If an image_url is provided, the model will use it as a base for editing.
    if image_url:
        print(f"🖼️  Using image from URL for editing: {image_url}")
        arguments["image_url"] = image_url

    try:
        # Call the new model endpoint "fal-ai/flux-pro/kontext/max"
        result = fal_client.subscribe(
            "fal-ai/flux-pro/kontext/max",
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        print("✅ Image generation successful!")
        return result
    except Exception as e:
        print(f"❌ An error occurred during image generation: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """
    The main execution block of the script.
    It parses command-line arguments to get the user's prompt and model selection, 
    calls the appropriate image generation function, and then downloads and saves 
    the resulting image to a local 'output' directory.
    """
    # argparse is used to create a command-line interface.
    parser = argparse.ArgumentParser(description="Generate an image from a text description using Fal-AI.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate the image from.")
    # New argument to select the model. Defaults to the original 'pony' model.
    parser.add_argument(
        "--model", 
        type=str, 
        default="pony", 
        choices=["pony", "kontext"], 
        help="The model to use for image generation. 'pony' for CyberRealistic Pony, 'kontext' for FLUX.1 Kontext."
    )
    # New argument for the image URL, used only by the 'kontext' model.
    parser.add_argument(
        "--image_url", 
        type=str, 
        help="URL of an image to use as a base for editing with the 'kontext' model."
    )
    # New argument for aspect ratio, used only by the 'kontext' model.
    parser.add_argument(
        "--aspect_ratio", 
        type=str, 
        default="1:1",
        choices=["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"],
        help="Aspect ratio for the 'kontext' model."
    )
    args = parser.parse_args()

    image_data = None
    # Based on the selected model, call the corresponding function.
    if args.model == "kontext":
        image_data = generate_image_with_kontext(args.prompt, args.image_url, args.aspect_ratio)
    else: # Default to pony
        image_data = generate_image_with_pony(args.prompt)

    # After generation, the 'image_data' dictionary should contain the URL of the image.
    if image_data and image_data.get('images'):
        image_url = image_data['images'][0]['url']
        response = requests.get(image_url)
        if response.status_code == 200:
            folder_path = "output"
            os.makedirs(folder_path, exist_ok=True)
            job_id = uuid.uuid4()
            filename = f"generated_image_{job_id}.png"
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Image saved as {file_path}")
            print(f"🔗 Image URL: {image_url}")
        else:
            print(f"❌ Failed to download image. Status code: {response.status_code}", file=sys.stderr)
    else:
        print("❌ No images were generated or the response was empty.", file=sys.stderr)

if __name__ == "__main__":
    main()
