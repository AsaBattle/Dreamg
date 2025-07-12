# create_lora_simple.py
# Simplified LoRA creation script for local use (no database required)

import argparse
import fal_client
import sys
import os
import shutil
import tempfile
from pathlib import Path

# Load Fal-AI API key from falkey.txt
try:
    with open("falkey.txt", "r") as f:
        os.environ["FAL_KEY"] = f.read().strip()
except FileNotFoundError:
    print("ERROR: falkey.txt not found. Please create this file and place your Fal API key in it.")
    sys.exit(1)

def on_queue_update(update):
    """
    Callback function to display logs during the LoRA creation process.
    This shows you real-time progress of the training happening on Fal.ai's servers.
    """
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

def upload_training_data(local_path: str) -> str:
    """
    Uploads a local directory (by zipping it) to fal.ai for training.
    
    Args:
        local_path: Path to the folder containing your training images
        
    Returns:
        URL of the uploaded zip file that Fal.ai can access for training
    """
    path = Path(local_path)
    if not path.exists() or not path.is_dir():
        print(f"❌ Error: The specified path is not a valid directory: {local_path}")
        sys.exit(1)

    # Check if there are images in the directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [f for f in path.iterdir() if f.suffix.lower() in image_extensions]
    
    if len(image_files) < 5:
        print(f"❌ Error: Need at least 5 images for training. Found {len(image_files)} images.")
        print("Add more .jpg, .png, or .webp files to your training folder.")
        sys.exit(1)
    
    print(f"📸 Found {len(image_files)} training images")

    temp_dir_to_clean = None
    try:
        print(f"📁 Zipping directory: {local_path}")
        temp_dir_to_clean = tempfile.mkdtemp()
        zip_path = Path(temp_dir_to_clean) / f"{path.name}.zip"
        shutil.make_archive(str(zip_path.with_suffix('')), 'zip', str(path))
        
        print(f"⬆️  Uploading '{os.path.basename(zip_path)}' to fal.ai...")
        image_url = fal_client.upload_file(str(zip_path))
        print(f"✅ Upload complete. URL: {image_url}")
        return image_url
    except Exception as e:
        print(f"❌ An error occurred during file upload: {e}")
        sys.exit(1)
    finally:
        if temp_dir_to_clean:
            shutil.rmtree(temp_dir_to_clean)

def create_lora_fal(images_url: str) -> dict:
    """
    Submits a request to the fal-ai API to create a new LoRA.
    
    Args:
        images_url: URL of the uploaded training images
        
    Returns:
        Dictionary containing the LoRA creation results including download URLs
    """
    print("🚀 Starting LoRA creation process on fal.ai...")
    print("⏰ This will take several minutes (usually 5-15 minutes)...")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": images_url, 
                "create_masks": True,  # Automatically create masks for better training
                "steps": 1000  # Number of training steps (1000 is good for most use cases)
            },
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        print("✅ LoRA creation successful!")
        return result
    except Exception as e:
        print(f"❌ An error occurred during LoRA creation: {e}")
        sys.exit(1)

def save_lora_info(lora_name: str, lora_data: dict):
    """
    Saves the LoRA information to a local text file for easy reference.
    
    Args:
        lora_name: Name you gave to the LoRA
        lora_data: The result data from Fal.ai containing URLs
    """
    try:
        # Create a 'loras' folder to store LoRA info
        loras_folder = Path("loras")
        loras_folder.mkdir(exist_ok=True)
        
        # Save LoRA info to a text file
        info_file = loras_folder / f"{lora_name}_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"LoRA Name: {lora_name}\n")
            f.write(f"Creation Date: {os.path.getctime}\n")
            f.write(f"LoRA File URL: {lora_data['diffusers_lora_file']['url']}\n")
            f.write(f"Config File URL: {lora_data['config_file']['url']}\n")
            f.write("\n--- Use this URL in your generate_image.py script ---\n")
            f.write(f"{lora_data['diffusers_lora_file']['url']}\n")
        
        print(f"📝 LoRA info saved to: {info_file}")
        print(f"🔗 Your LoRA URL: {lora_data['diffusers_lora_file']['url']}")
        print(f"💡 Copy this URL to use in your image generation script!")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save LoRA info to file: {e}")
        print(f"🔗 Your LoRA URL: {lora_data['diffusers_lora_file']['url']}")
        print("Make sure to copy this URL - you'll need it for image generation!")

def main():
    """
    Main function to parse arguments, create a LoRA, and save the results locally.
    """
    parser = argparse.ArgumentParser(description="Create a new LoRA from local images.")
    parser.add_argument("--name", type=str, required=True, 
                       help="A friendly name for your LoRA (e.g., 'john_smith', 'my_character')")
    parser.add_argument("--path", type=str, required=True, 
                       help="Path to the folder containing your training images")
    args = parser.parse_args()

    print(f"🎯 Creating LoRA '{args.name}' from images in '{args.path}'")
    
    # Upload the training images
    images_url = upload_training_data(args.path)
    
    # Create the LoRA
    lora_data = create_lora_fal(images_url)

    if lora_data:
        # Save the LoRA information locally
        save_lora_info(args.name, lora_data)
        print(f"🎉 LoRA '{args.name}' has been created successfully!")
        print("📋 Next steps:")
        print("1. Copy the LoRA URL from the file created")
        print("2. Modify your generate_image.py script to include this LoRA")
        print("3. Start generating images with your custom LoRA!")
    else:
        print("❌ LoRA creation failed - no data returned")

if __name__ == "__main__":
    main() 