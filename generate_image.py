We need to make sure the images being generated in the create_lora function are good and properly following the prompt completely!(The lora made last created a lady with blue eyes but the prompt said brown)

# generate_image.py
# This script uses the fal-ai API to generate an image from a user-provided text prompt,
# generate a dataset of images for LoRA training, or run the full end-to-end LoRA creation pipeline.

import argparse
import fal_client
import sys
import os
import requests
import uuid
import tempfile
import zipfile
import shutil

# The various activities for Kontext to use for dataset generation.
# Each string in this list will be used as a prompt.
ACTIVITY_PROMPTS = [
    "A photo of the person sitting at a table, having coffee, smiling at the camera, photorealistic, 8k",
    "A photo of the person skiing on a mountain, having fun, with a trying hard look on their face, action shot, photorealistic, 8k",
    "A photo of the person running in a park, with exercise-related clothing on, not looking at the camera, candid, photorealistic, 8k",
    "A photo of the person sitting on a park bench, looking toward the sky, slightly smiling, peaceful, photorealistic, 8k",
    "A photo of the person yelling at someone just out of frame, looking angry, dramatic lighting, photorealistic, 8k"
]

# Load Fal-AI API key from falkey.txt
try:
    with open("falkey.txt", "r") as f:
        os.environ["FAL_KEY"] = f.read().strip()
except FileNotFoundError:
    print("ERROR: falkey.txt not found. Please create this file and place your Fal API key in it.")
    sys.exit(1)


def on_queue_update(update):
    """Callback to print real-time logs from the image generation process."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

# --- Core Generation Functions ---

def generate_image_with_pony(prompt: str) -> list:
    """Generates a single image using the CyberRealistic Pony model from a text prompt."""
    print("🎨 Starting image generation with CyberRealistic Pony model...")
    full_prompt = f"score_9, score_8_up, score_7_up, BREAK, ((score_9, score_8_up, score_7_up, epic, gorgeous, film grain, grainy, {prompt}, detailed face, (detailed skin pores:1.1), film grain, low light, detailed eyes, detailed skin, (photorealistic:1.1), (amateur photography:1.1)"
    try:
        result = fal_client.subscribe(
            "fal-ai/lora",
            arguments={
                "prompt": full_prompt,
                "model_name": "John6666/cyberrealistic-pony-v8-sdxl",
                "enable_safety_checker": False,
                "clip_skip": 2,
                "scheduler": "DPM++ 2M SDE Karras",
                "image_size": {"width": 1024, "height": 1024},
                "num_images": 1,
                "image_format": "png",
                "guidance_scale": 5,
                "negative_prompt": "score_1, score_2, score_3, text, watermarks, child, youth, underage, teddy bears, stuffed animals, ",
                "prompt_weighting": True,
                "num_inference_steps": 30
            },
            with_logs=True,
            on_queue_update=on_queue_update
        )
        print("✅ Pony image generation successful!")
        return [result]
    except Exception as e:
        print(f"❌ An error occurred during Pony image generation: {e}", file=sys.stderr)
        sys.exit(1)

def generate_with_lora(prompt: str, lora_url: str) -> list:
    """Generates a single image using the Pony model with a specified LoRA file."""
    print(f"🎨 Starting image generation with Pony model and LoRA: {lora_url}...")
    full_prompt = f"score_9, score_8_up, score_7_up, BREAK, ((score_9, score_8_up, score_7_up, epic, gorgeous, film grain, grainy, {prompt}, detailed face, (detailed skin pores:1.1), film grain, low light, detailed eyes, detailed skin, (photorealistic:1.1), (amateur photography:1.1)"
    try:
        result = fal_client.subscribe(
            "fal-ai/lora",
            arguments={
                "prompt": full_prompt,
                "model_name": "John6666/cyberrealistic-pony-v8-sdxl",
                "lora": lora_url,
                "lora_scale": 0.8,
                "enable_safety_checker": False,
                "clip_skip": 2,
                "scheduler": "DPM++ 2M SDE Karras",
                "image_size": {"width": 1024, "height": 1024},
                "num_images": 1,
                "image_format": "png",
                "guidance_scale": 5,
                "negative_prompt": "score_1, score_2, score_3, text, watermarks, child, youth, underage, teddy bears, stuffed animals, ",
                "prompt_weighting": True,
                "num_inference_steps": 30
            },
            with_logs=True,
            on_queue_update=on_queue_update
        )
        print("✅ LoRA-based image generation successful!")
        return [result]
    except Exception as e:
        print(f"❌ An error occurred during LoRA-based image generation: {e}", file=sys.stderr)
        sys.exit(1)

def generate_single_image_with_flux(prompt: str, aspect_ratio: str = "1:1") -> list:
    """Generates a single image using the FLUX.1 Pro Text-to-Image model."""
    print("🎨 Starting single image generation with FLUX.1 Pro Text-to-Image model...")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-pro",
            arguments={
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "enable_safety_checker": False,
                "safety_tolerance": "6",
                "guidance_scale": 3.5,
                "num_images": 1,
                "output_format": "png",
            },
            with_logs=True,
            on_queue_update=on_queue_update
        )
        print("✅ Flux single image generation successful!")
        return [result]
    except Exception as e:
        print(f"❌ An error occurred during Flux single image generation: {e}", file=sys.stderr)
        sys.exit(1)

def generate_activities_with_flux(image_url: str, aspect_ratio: str = "1:1") -> list:
    """Generates a dataset of images by placing a person into various activities."""
    print(f"🎨 Starting dataset generation with FLUX.1 Kontext [Max] model for {len(ACTIVITY_PROMPTS)} activities...")
    all_results = []
    for i, activity_prompt in enumerate(ACTIVITY_PROMPTS):
        print(f"\n--- Generating activity {i+1}/{len(ACTIVITY_PROMPTS)}: '{activity_prompt}' ---")
        try:
            result = fal_client.subscribe(
                "fal-ai/flux-pro/kontext/max",
                arguments={
                    "prompt": activity_prompt,
                    "image_url": image_url,
                    "aspect_ratio": aspect_ratio,
                    "guidance_scale": 3.5,
                    "safety_tolerance": "6",
                    "num_images": 1,
                    "output_format": "png",
                },
                with_logs=True,
                on_queue_update=on_queue_update
            )
            all_results.append(result)
            print(f"✅ Activity {i+1} successful!")
        except Exception as e:
            print(f"❌ An error occurred generating activity {i+1}: {e}", file=sys.stderr)
            continue
    print("\n✅ All activities processed!")
    return all_results

def train_lora_model(zip_url: str, trigger_word: str) -> dict:
    """Starts and monitors a LoRA training job on fal.ai using the FLUX training model."""
    print(f"Submitting training job to fal-ai/flux-lora-general-training with trigger: '{trigger_word}'")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-lora-general-training",
            arguments={
                "images_data_url": zip_url,
                "trigger_word": trigger_word
            },
            with_logs=True,
            on_queue_update=on_queue_update
        )
        return result
    except Exception as e:
        print(f"❌ An error occurred during LoRA training: {e}", file=sys.stderr)
        return {}

def create_lora_workflow(prompt: str, trigger_word: str):
    """Automates the full pipeline from a text prompt to a trained LoRA file."""
    print("--- STEP 1/5: Generating base image from prompt ---")
    base_image_results = generate_single_image_with_flux(prompt)
    if not base_image_results or not base_image_results[0].get('images'):
        print("❌ Failed to generate base image. Aborting workflow.", file=sys.stderr)
        return
    base_image_url = base_image_results[0]['images'][0]['url']
    print(f"✅ Base image generated: {base_image_url}")

    print("\n--- STEP 2/5: Generating activity dataset from base image ---")
    activity_results = generate_activities_with_flux(base_image_url)
    if not activity_results:
        print("❌ Failed to generate activity images. Aborting workflow.", file=sys.stderr)
        return
    
    dataset_image_urls = [res['images'][0]['url'] for res in activity_results if res and res.get('images')]
    dataset_image_urls.insert(0, base_image_url)
    print(f"✅ Generated {len(dataset_image_urls)} total images for the dataset.")

    print("\n--- STEP 3/5: Preparing and zipping dataset ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, url in enumerate(dataset_image_urls):
            try:
                response = requests.get(url)
                response.raise_for_status()
                with open(os.path.join(tmpdir, f"image_{i+1}.png"), "wb") as f: f.write(response.content)
            except requests.RequestException as e:
                print(f"⚠️ Could not download {url}: {e}. Skipping.", file=sys.stderr)
                continue
        print(f"✅ Downloaded {len(os.listdir(tmpdir))} images to temporary directory.")
        zip_path = shutil.make_archive(os.path.join(os.getcwd(), f"lora_dataset_{uuid.uuid4()}"), 'zip', tmpdir)
        print(f"✅ Dataset zipped to: {zip_path}")

    print("\n--- STEP 4/5: Uploading zipped dataset ---")
    try:
        training_data_url = fal_client.upload_file(zip_path)
        print(f"✅ Dataset uploaded: {training_data_url}")
    except Exception as e:
        print(f"❌ Failed to upload zip file: {e}. Aborting.", file=sys.stderr)
        os.remove(zip_path)
        return
    finally:
        os.remove(zip_path)

    print("\n--- STEP 5/5: Starting LoRA training job ---")
    training_.result = train_lora_model(training_data_url, trigger_word)
    if training_result and training_result.get('diffusers_lora_file'):
        lora_url = training_result['diffusers_lora_file']['url']
        print(f"\n✅ Training complete! Downloading LoRA file from: {lora_url}")
        os.makedirs("loras", exist_ok=True)
        # UPDATED: Save with trigger word in the filename
        lora_path = os.path.join("loras", f"lora_{trigger_word}_{uuid.uuid4()}.zip")
        try:
            response = requests.get(lora_url)
            response.raise_for_status()
            with open(lora_path, "wb") as f: f.write(response.content)
            print(f"✅ LoRA model saved to: {lora_path}")
        except requests.RequestException as e:
            print(f"❌ Failed to download LoRA file: {e}", file=sys.stderr)
    else:
        print("❌ Training job did not return a LoRA file.", file=sys.stderr)

def main():
    """Main execution block. Parses CLI arguments and calls the correct function."""
    parser = argparse.ArgumentParser(description="A tool for image generation and LoRA creation with Fal-AI.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="The command to run.")

    pony_parser = subparsers.add_parser('pony', help='Generate a single image with the Pony model.')
    pony_parser.add_argument('prompt', type=str, help='The text prompt.')

    flux_parser = subparsers.add_parser('flux', help='Generate with the FLUX model.')
    flux_group = flux_parser.add_mutually_exclusive_group(required=True)
    flux_group.add_argument('--prompt', type=str, help='Text prompt for single image generation.')
    flux_group.add_argument('--image_url', type=str, help='URL or local file for dataset creation.')
    flux_parser.add_argument('--aspect_ratio', type=str, default="1:1", choices=["21:9", "16:9", "4:3", "3:2", "1:1", "2:3", "3:4", "9:16", "9:21"])
    
    create_lora_parser = subparsers.add_parser('create_lora', help='End-to-end: Create a FLUX LoRA from a prompt.')
    create_lora_parser.add_argument('prompt', type=str, help='A detailed description of the person.')
    # UPDATED: Added required trigger_word argument
    create_lora_parser.add_argument('--trigger_word', required=True, type=str, help='A unique word to trigger the LoRA (e.g., ohwxman).')

    use_lora_parser = subparsers.add_parser('use_lora', help='Generate an image with the Pony model using a specific LoRA.')
    use_lora_parser.add_argument('--prompt', required=True, type=str, help='The text prompt for the image.')
    use_lora_parser.add_argument('--lora_path', required=True, type=str, help='URL or local file path of the LoRA model.')
    
    args = parser.parse_args()

    file_to_upload = None
    if args.command == 'flux' and getattr(args, 'image_url', None):
        file_to_upload = args.image_url
    elif args.command == 'use_lora':
        file_to_upload = args.lora_path

    final_url = file_to_upload
    if file_to_upload and not file_to_upload.startswith(('http://', 'https://')):
        if os.path.exists(file_to_upload):
            try:
                print(f"📦 Uploading local file: {file_to_upload}")
                final_url = fal_client.upload_file(file_to_upload)
                print(f"🔗 Upload successful. Public URL: {final_url}")
            except Exception as e:
                print(f"❌ Failed to upload file: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"❌ Local file not found at path: {file_to_upload}", file=sys.stderr)
            sys.exit(1)
    
    results = None
    if args.command == 'create_lora':
        # UPDATED: Pass the trigger word to the workflow
        create_lora_workflow(args.prompt, args.trigger_word)
        return
    elif args.command == 'pony':
        results = generate_image_with_pony(args.prompt)
    elif args.command == 'flux':
        if args.prompt:
            results = generate_single_image_with_flux(args.prompt, args.aspect_ratio)
        elif final_url:
            results = generate_activities_with_flux(final_url, args.aspect_ratio)
    elif args.command == 'use_lora':
        results = generate_with_lora(args.prompt, final_url)

    if results:
        print(f"\n⬇️  Downloading {len(results)} generated image(s)...")
        os.makedirs("output", exist_ok=True)
        job_id = uuid.uuid4()
        for i, image_data in enumerate(results):
            if image_data and image_data.get('images'):
                try:
                    image_url = image_data['images'][0]['url']
                    response = requests.get(image_url)
                    response.raise_for_status()
                    
                    is_dataset = args.command == 'flux' and getattr(args, 'image_url', None)
                    filename = f"generated_image_{job_id}_activity_{i+1}.png" if is_dataset else f"generated_image_{job_id}.png"
                    file_path = os.path.join("output", filename)

                    with open(file_path, "wb") as f: f.write(response.content)
                    print(f"✅ Image {i+1} saved as {file_path}\n   🔗 URL: {image_url}")
                except (requests.RequestException, KeyError) as e:
                    print(f"❌ Failed to process result {i+1}. Error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
