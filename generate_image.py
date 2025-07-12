# generate_image.py
# This script uses the fal-ai API to generate an image from a user-provided text prompt.

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

def generate_image(prompt: str) -> dict:
    """
    This function sends a request to the fal-ai API to generate an image using the CyberRealistic Pony model.
    It takes a text prompt and uses the 'fal-ai/lora' endpoint with a specific SDXL-based model that's designed
    for photorealistic image generation. The prompt is automatically formatted with the special tokens and 
    structure required by Pony Diffusion models.

    Args:
        prompt: The text description of the image to be generated (this will replace "a beautiful lady" in the template).

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
                "clip_skip": 2,  # This setting helps with Pony models for better prompt understanding
                "scheduler": "DPM++ 2M SDE Karras",  # A high-quality sampling method
                "image_size": {
                    "width": 1024,
                    "height": 1024
                },
                "model_name": "John6666/cyberrealistic-pony-v8-sdxl",  # The specific CyberRealistic Pony model
                "num_images": 1,
                "image_format": "png",  # PNG format for better quality
                "guidance_scale": 5,  # Controls how closely the model follows the prompt
                "enable_safety_checker": False,  # Disabled for more creative freedom
                # Negative prompt helps exclude unwanted elements from the generation
                "negative_prompt": "score_1, score_2, score_3, text, watermarks, child, youth, underage, teddy bears, stuffed animals, ",
                "prompt_weighting": True,  # Allows for weighted prompt terms using parentheses
                "num_inference_steps": 30  # Number of denoising steps for quality
            },
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
    It parses command-line arguments to get the user's prompt, calls the image generation function,
    and then downloads and saves the resulting image to a local 'output' directory.
    """
    # argparse is used to create a command-line interface.
    # Here, we define one required argument: 'prompt'.
    parser = argparse.ArgumentParser(description="Generate an image from a text description using Fal-AI.")
    parser.add_argument("prompt", type=str, help="The text prompt to generate the image from.")
    args = parser.parse_args()

    # Call the function to start the image generation process.
    image_data = generate_image(args.prompt)

    # After generation, the 'image_data' dictionary should contain the URL of the image.
    if image_data and image_data.get('images'):
        image_url = image_data['images'][0]['url']

        # The requests library is used to download the image from the URL.
        response = requests.get(image_url)
        if response.status_code == 200:
            # The image will be saved in an 'output' folder.
            # os.makedirs ensures the directory exists without raising an error if it's already there.
            folder_path = "output"
            os.makedirs(folder_path, exist_ok=True)

            # A unique filename is generated using UUID to prevent overwriting previous images.
            job_id = uuid.uuid4()
            filename = f"generated_image_{job_id}.png"
            file_path = os.path.join(folder_path, filename)

            # The image content is written to a new file in binary mode.
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Image saved as {file_path}")
            print(f"🔗 Image URL: {image_url}")
        else:
            print(f"❌ Failed to download image. Status code: {response.status_code}", file=sys.stderr)
    else:
        print("❌ No images were generated or the response was empty.", file=sys.stderr)

# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly from the command line.
if __name__ == "__main__":
    main() 