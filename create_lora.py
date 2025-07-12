# create_lora.py
# Newest versiopnm of this file

import argparse
import sqlite3
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

# Use the same database file as your Discord bot
DB_FILE = "/home/web1/ernbot/fluxloras.db"

def setup_database():
    """Initializes the SQLite database and creates the loras table if it doesn't exist."""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # Updated schema to include userid and match your bot's conventions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS loras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                userid TEXT NOT NULL,
                friendly_name TEXT NOT NULL,
                lora_url TEXT NOT NULL,
                config_url TEXT,
                newurl TEXT,
                UNIQUE(userid, friendly_name)
            )
        """)
        conn.commit()

def on_queue_update(update):
    """Callback function to display logs during the LoRA creation process."""
    if isinstance(update, fal_client.InProgress):
        for log in update.logs:
            print(log["message"])

def upload_training_data(local_path: str) -> str:
    """Uploads a local directory (by zipping it) to fal.ai."""
    path = Path(local_path)
    if not path.exists() or not path.is_dir():
        print(f"❌ Error: The specified path is not a valid directory: {local_path}")
        sys.exit(1)

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
    """Submits a request to the fal-ai API to create a new LoRA."""
    print("🚀 Starting LoRA creation process on fal.ai...")
    try:
        result = fal_client.subscribe(
            "fal-ai/flux-lora-fast-training",
            arguments={"images_data_url": images_url, "create_masks": True, "steps": 1000},
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        print("✅ LoRA creation successful!")
        return result
    except Exception as e:
        print(f"❌ An error occurred during LoRA creation: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments, create a LoRA, and save it to the database."""
    parser = argparse.ArgumentParser(description="Create a new LoRA from local images and store it.")
    parser.add_argument("--name", type=str, required=True, help="A unique friendly name for the new LoRA.")
    parser.add_argument("--path", type=str, required=True, help="The local path to the directory of images.")
    parser.add_argument("--userid", type=str, required=True, help="The Discord user ID of the creator.")
    args = parser.parse_args()

    setup_database()

    # Set Fal credentials from environment variables
    # Note: Ensure these are set in the environment where your bot runs.
#    if not os.getenv("FAL_KEY_ID") or not os.getenv("FAL_KEY_SECRET"):
#        print("❌ Environment variables FAL_KEY_ID and FAL_KEY_SECRET must be set.")
#        sys.exit(1)
    
    images_url = upload_training_data(args.path)
    lora_data = create_lora_fal(images_url)

    if lora_data:
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO loras (userid, friendly_name, lora_url, config_url, newurl) VALUES (?, ?, ?, ?, ?)",
                    (
                        args.userid,
                        args.name,
                        lora_data['diffusers_lora_file']['url'],
                        lora_data['config_file']['url'],
                        lora_data['diffusers_lora_file']['url'] # Storing main URL in newurl for compatibility
                    )
                )
                conn.commit()
            print(f"🎉 LoRA '{args.name}' for user {args.userid} has been saved to the database.")
        except sqlite3.IntegrityError:
            print(f"⚠️ A LoRA with the name '{args.name}' already exists for this user.")
            # Optionally, you could update the existing entry here
        except Exception as e:
            print(f"❌ An error occurred while saving to the database: {e}")

if __name__ == "__main__":
    main()