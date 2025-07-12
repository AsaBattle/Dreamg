# gen_with_lora.py (Updated for multi-user bot)

import argparse
import sqlite3
import fal_client
import sys
import os
import requests
from PIL import Image
import io

# Load Fal-AI API key from falkey.txt
try:
    with open("falkey.txt", "r") as f:
        os.environ["FAL_KEY"] = f.read().strip()
except FileNotFoundError:
    print("ERROR: falkey.txt not found. Please create this file and place your Fal API key in it.")
    sys.exit(1)


# --- Database Interaction ---
# Make sure this points to your bot's central database
DB_FILE = "/home/web1/ernbot/fluxloras.db" 

# UPDATED: Now requires a userid to find the correct LoRA
def get_lora_url(friendly_name: str, userid: str) -> str:
    """
    Retrieves the LoRA file URL for a specific user.
    """
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            # Use the 'newurl' column to be consistent with your other commands
            cursor.execute("SELECT newurl FROM loras WHERE friendly_name = ? AND userid = ?", (friendly_name, userid))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                return None # Return None if not found, so the bot can handle the error
    except Exception as e:
        print(f"❌ An error occurred while accessing the database: {e}", file=sys.stderr)
        sys.exit(1)

# --- Fal Client Interaction ---
def on_queue_update(update):
    """Callback function to display logs during the image generation process."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

def generate_image_with_lora(prompt: str, lora_url: str) -> dict:
    """
    Submits a request to the fal-ai API to generate an image using a specific LoRA.

    Args:
        prompt: The text prompt for the image generation.
        lora_url: The URL of the LoRA file to apply.

    Returns:
        A dictionary containing the generated image information.
    """
    print("🎨 Starting image generation...")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-lora",
            arguments={
                "prompt": prompt,
                "loras": [{
                    "path": lora_url,
                    "scale": 1
                }],
                "image_size": "landscape_4_3",
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "num_images": 1,
                "enable_safety_checker": False,
                "output_format": "jpeg"
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        print("✅ Image generation successful!")
        return result
    except Exception as e:
        print(f"❌ An error occurred during image generation: {e}")
        sys.exit(1)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate an image using a specified LoRA.")
    parser.add_argument("friendly_name", type=str, help="The friendly name of the LoRA to use.")
    parser.add_argument("prompt", type=str, help="The prompt to generate the image from.")
    parser.add_argument("job_id", type=str, help="The job ID for tracking.")
    # ADDED: Required userid argument
    parser.add_argument("--userid", type=str, required=True, help="The Discord user ID for LoRA lookup.")
    args = parser.parse_args()

    # It's better to rely on environment variables set when the bot starts
#    if not os.getenv("FAL_KEY_ID") or not os.getenv("FAL_KEY_SECRET"):
#        print("❌ Environment variables FAL_KEY_ID and FAL_KEY_SECRET must be set.", file=sys.stderr)
#        sys.exit(1)

    # UPDATED: Pass userid to the lookup function
    lora_url = get_lora_url(args.friendly_name, args.userid)
    if not lora_url:
        print(f"❌ LoRA '{args.friendly_name}' not found for user {args.userid}.", file=sys.stderr)
        sys.exit(1)

    image_data = generate_image_with_lora(args.prompt, lora_url)

    if image_data and image_data.get('images'):
        image_url = image_data['images'][0]['url']
        response = requests.get(image_url)
        if response.status_code == 200:
            folder_path = "output"
            os.makedirs(folder_path, exist_ok=True)
            filename = f"{args.friendly_name}_{args.job_id}_generated_image.jpg"
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Image saved as {file_path}")
        else:
            print(f"❌ Failed to download image. Status code: {response.status_code}", file=sys.stderr)
    else:
        print("❌ No images were generated.", file=sys.stderr)

if __name__ == "__main__":
    main()