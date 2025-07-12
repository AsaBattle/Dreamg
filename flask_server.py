# flask_server.py
# Flask API server for image generation and LoRA creation

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import fal_client
import sys
import os
import requests
import uuid
import shutil
import tempfile
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
import base64
import io
from PIL import Image
import traceback

# Load Fal-AI API key from falkey.txt
try:
    with open("falkey.txt", "r") as f:
        os.environ["FAL_KEY"] = f.read().strip()
except FileNotFoundError:
    print("ERROR: falkey.txt not found. Please create this file and place your Fal API key in it.")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (allows C# to call the API)

# Global variables for tracking ongoing operations
active_jobs = {}
job_lock = threading.Lock()

def generate_job_id():
    """Generate a unique job ID for tracking operations"""
    return str(uuid.uuid4())

def on_queue_update_factory(job_id):
    """Factory function to create a callback for specific job tracking"""
    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['logs'].extend([log["message"] for log in update.logs])
                    active_jobs[job_id]['last_update'] = datetime.now()
    return on_queue_update

def download_image(image_url: str, output_dir: str = "output") -> str:
    """
    Download an image from URL and save it locally
    
    Args:
        image_url: URL of the image to download
        output_dir: Directory to save the image
        
    Returns:
        Local file path of the downloaded image
    """
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            os.makedirs(output_dir, exist_ok=True)
            job_id = uuid.uuid4()
            filename = f"generated_image_{job_id}.png"
            file_path = os.path.join(output_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        else:
            raise Exception(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"Error downloading image: {str(e)}")

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/generate-image', methods=['POST'])
def generate_image_endpoint():
    """
    Generate an image from a text prompt
    
    Expected JSON payload:
    {
        "prompt": "description of the image",
        "lora_url": "optional_lora_url",
        "lora_scale": 1.0,
        "return_base64": false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request', 'success': False}), 400
        
        prompt = data['prompt']
        lora_url = data.get('lora_url')
        lora_scale = data.get('lora_scale', 1.0)
        return_base64 = data.get('return_base64', False)
        
        # Create job ID for tracking
        job_id = generate_job_id()
        
        # Initialize job tracking
        with job_lock:
            active_jobs[job_id] = {
                'type': 'image_generation',
                'status': 'processing',
                'logs': [],
                'created': datetime.now(),
                'last_update': datetime.now()
            }
        
        # Create the full prompt with Pony Diffusion formatting
        full_prompt = f"score_9, score_8_up, score_7_up, BREAK, ((score_9, score_8_up, score_7_up, epic, gorgeous, film grain, grainy, {prompt}, detailed face, (detailed skin pores:1.1), film grain, low light, detailed eyes, detailed skin, (photorealistic:1.1), (amateur photography:1.1)"
        
        # Prepare arguments for Fal.ai API
        arguments = {
            "prompt": full_prompt,
            "clip_skip": 2,
            "scheduler": "DPM++ 2M SDE Karras",
            "image_size": {
                "width": 1024,
                "height": 1024
            },
            "model_name": "John6666/cyberrealistic-pony-v8-sdxl",
            "num_images": 1,
            "image_format": "png",
            "guidance_scale": 5,
            "enable_safety_checker": False,
            "negative_prompt": "score_1, score_2, score_3, text, watermarks, child, youth, underage, teddy bears, stuffed animals, ",
            "prompt_weighting": True,
            "num_inference_steps": 30
        }
        
        # Add LoRA if provided
        if lora_url:
            arguments["loras"] = [{
                "path": lora_url,
                "scale": lora_scale
            }]
        
        # Generate image
        result = fal_client.subscribe(
            "fal-ai/lora",
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update_factory(job_id),
        )
        
        if result and result.get('images'):
            image_url = result['images'][0]['url']
            
            # Update job status
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'completed'
                    active_jobs[job_id]['image_url'] = image_url
            
            response_data = {
                'success': True,
                'job_id': job_id,
                'image_url': image_url,
                'prompt_used': full_prompt
            }
            
            # If base64 is requested, download and convert the image
            if return_base64:
                try:
                    img_response = requests.get(image_url)
                    if img_response.status_code == 200:
                        img_base64 = base64.b64encode(img_response.content).decode('utf-8')
                        response_data['image_base64'] = img_base64
                    else:
                        error_message = f"Failed to download image for base64 conversion. Status: {img_response.status_code}, Body: {img_response.text}"
                        app.logger.error(error_message)
                        response_data['base64_error'] = error_message
                except Exception as e:
                    error_details = traceback.format_exc()
                    app.logger.error(f"Error during base64 conversion: {error_details}")
                    response_data['base64_error'] = f"An unexpected error occurred during base64 conversion: {str(e)}"
            
            return jsonify(response_data)
        else:
            # Update job status
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'failed'
                    active_jobs[job_id]['error'] = f'No images generated. Result: {result}'
            
            error_message = 'No images were generated.'
            details = f"The result from the generation service was empty or did not contain images. Full result: {result}"
            app.logger.error(f"{error_message} {details}")
            return jsonify({
                'success': False,
                'error': error_message,
                'details': details,
                'job_id': job_id
            }), 500
            
    except Exception as e:
        error_details = traceback.format_exc()
        app.logger.error(f"Error in /generate-image endpoint: {error_details}")

        # Update job status if job_id exists
        if 'job_id' in locals():
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'failed'
                    active_jobs[job_id]['error'] = str(e)
                    active_jobs[job_id]['traceback'] = error_details
        
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred in generate_image_endpoint: {str(e)}',
            'traceback': error_details
        }), 500

@app.route('/create-lora', methods=['POST'])
def create_lora_endpoint():
    """
    Create a LoRA from uploaded images
    
    Expected JSON payload:
    {
        "name": "lora_name",
        "images": ["base64_image1", "base64_image2", ...],
        "steps": 1000
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'name' not in data or 'images' not in data:
            return jsonify({'error': 'Missing name or images in request', 'success': False}), 400
        
        lora_name = data['name']
        images_base64 = data['images']
        steps = data.get('steps', 1000)
        
        if len(images_base64) < 5:
            return jsonify({'error': 'Need at least 5 images for LoRA training', 'success': False}), 400
        
        # Create job ID for tracking
        job_id = generate_job_id()
        
        # Initialize job tracking
        with job_lock:
            active_jobs[job_id] = {
                'type': 'lora_creation',
                'status': 'processing',
                'logs': [],
                'created': datetime.now(),
                'last_update': datetime.now()
            }
        
        # Create temporary directory for images
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Decode and save images
            for i, img_base64 in enumerate(images_base64):
                try:
                    # Remove data URL prefix if present
                    if img_base64.startswith('data:'):
                        img_base64 = img_base64.split(',')[1]
                    
                    # Decode base64
                    img_data = base64.b64decode(img_base64)
                    
                    # Save image
                    img_path = os.path.join(temp_dir, f"image_{i+1}.png")
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                        
                except Exception as e:
                    error_details = traceback.format_exc()
                    app.logger.error(f"Failed to process image {i+1}: {error_details}")
                    return jsonify({'error': f'Failed to process image {i+1}: {str(e)}', 'success': False, 'traceback': error_details}), 400
            
            # Create zip file for upload
            zip_path = os.path.join(tempfile.gettempdir(), f"{lora_name}_{job_id}.zip")
            shutil.make_archive(zip_path.replace('.zip', ''), 'zip', temp_dir)
            
            # Upload to Fal.ai
            images_url = fal_client.upload_file(zip_path)
            
            # Update job status
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['logs'].append(f"Images uploaded: {images_url}")
            
            # Create LoRA
            result = fal_client.subscribe(
                "fal-ai/flux-lora-fast-training",
                arguments={
                    "images_data_url": images_url,
                    "create_masks": True,
                    "steps": steps
                },
                with_logs=True,
                on_queue_update=on_queue_update_factory(job_id),
            )
            
            if result:
                lora_url = result['diffusers_lora_file']['url']
                config_url = result['config_file']['url']
                
                # Save LoRA info locally
                loras_folder = Path("loras")
                loras_folder.mkdir(exist_ok=True)
                
                lora_info = {
                    'name': lora_name,
                    'created': datetime.now().isoformat(),
                    'lora_url': lora_url,
                    'config_url': config_url,
                    'steps': steps,
                    'job_id': job_id
                }
                
                info_file = loras_folder / f"{lora_name}_info.json"
                with open(info_file, 'w') as f:
                    json.dump(lora_info, f, indent=2)
                
                # Update job status
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id]['status'] = 'completed'
                        active_jobs[job_id]['lora_url'] = lora_url
                        active_jobs[job_id]['config_url'] = config_url
                
                return jsonify({
                    'success': True,
                    'job_id': job_id,
                    'lora_name': lora_name,
                    'lora_url': lora_url,
                    'config_url': config_url
                })
            else:
                # Update job status
                with job_lock:
                    if job_id in active_jobs:
                        active_jobs[job_id]['status'] = 'failed'
                        active_jobs[job_id]['error'] = f"LoRA creation failed. Result: {result}"
                
                error_message = 'LoRA creation failed - no data returned from training service.'
                details = f"Fal-ai result: {result}"
                app.logger.error(f"{error_message} {details}")
                return jsonify({
                    'success': False,
                    'error': error_message,
                    'details': details,
                    'job_id': job_id
                }), 500
                
        finally:
            # Clean up temporary files
            try:
                shutil.rmtree(temp_dir)
                if os.path.exists(zip_path):
                    os.remove(zip_path)
            except:
                pass
            
    except Exception as e:
        error_details = traceback.format_exc()
        app.logger.error(f"Error in /create-lora endpoint: {error_details}")

        # Update job status if job_id exists
        if 'job_id' in locals():
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'failed'
                    active_jobs[job_id]['error'] = str(e)
                    active_jobs[job_id]['traceback'] = error_details
        
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred in create_lora_endpoint: {str(e)}',
            'traceback': error_details
        }), 500

@app.route('/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get the status of a specific job"""
    with job_lock:
        if job_id in active_jobs:
            job_info = active_jobs[job_id].copy()
            # Convert datetime objects to strings
            job_info['created'] = job_info['created'].isoformat()
            job_info['last_update'] = job_info['last_update'].isoformat()
            return jsonify({
                'success': True,
                'job_id': job_id,
                'job_info': job_info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

@app.route('/list-loras', methods=['GET'])
def list_loras():
    """List all created LoRAs"""
    try:
        loras_folder = Path("loras")
        if not loras_folder.exists():
            return jsonify({
                'success': True,
                'loras': []
            })
        
        loras = []
        for info_file in loras_folder.glob("*_info.json"):
            try:
                with open(info_file, 'r') as f:
                    lora_info = json.load(f)
                    loras.append(lora_info)
            except:
                continue
        
        return jsonify({
            'success': True,
            'loras': loras
        })
        
    except Exception as e:
        error_details = traceback.format_exc()
        app.logger.error(f"Error in /list-loras endpoint: {error_details}")
        return jsonify({
            'success': False,
            'error': f'An unexpected error occurred in list_loras: {str(e)}',
            'traceback': error_details
        }), 500

@app.route('/download-image/<job_id>', methods=['GET'])
def download_image(job_id):
    """Download the generated image for a specific job"""
    try:
        with job_lock:
            if job_id not in active_jobs:
                return jsonify({'error': 'Job not found', 'success': False}), 404
            
            job_info = active_jobs[job_id]
            if job_info['status'] != 'completed' or 'image_url' not in job_info:
                return jsonify({'error': 'Image not available', 'success': False}), 404
            
            image_url = job_info['image_url']
        
        # Download image
        response = requests.get(image_url)
        if response.status_code == 200:
            return send_file(
                io.BytesIO(response.content),
                mimetype='image/png',
                as_attachment=True,
                download_name=f'generated_image_{job_id}.png'
            )
        else:
            error_message = f"Failed to download image. Status: {response.status_code}, Body: {response.text}"
            app.logger.error(error_message)
            return jsonify({'error': 'Failed to download image', 'details': error_message, 'success': False}), 500
            
    except Exception as e:
        error_details = traceback.format_exc()
        app.logger.error(f"Error in /download-image/{job_id} endpoint: {error_details}")
        return jsonify({'error': f"An unexpected error occurred: {str(e)}", 'success': False, 'traceback': error_details}), 500

# Cleanup old jobs (runs every hour)
def cleanup_old_jobs():
    """Remove jobs older than 24 hours"""
    while True:
        try:
            time.sleep(3600)  # Sleep for 1 hour
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            with job_lock:
                jobs_to_remove = []
                for job_id, job_info in active_jobs.items():
                    if job_info['created'] < cutoff_time:
                        jobs_to_remove.append(job_id)
                
                for job_id in jobs_to_remove:
                    del active_jobs[job_id]
                    
        except Exception as e:
            app.logger.error(f"Error in cleanup_old_jobs: {traceback.format_exc()}")

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
    cleanup_thread.start()
    
    # Start Flask server
    print("🚀 Starting Flask API server...")
    print("📋 Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /generate-image - Generate image from prompt")
    print("  POST /create-lora - Create LoRA from images")
    print("  GET  /job-status/<job_id> - Get job status")
    print("  GET  /list-loras - List all created LoRAs")
    print("  GET  /download-image/<job_id> - Download generated image")
    print("🌐 Server running on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 