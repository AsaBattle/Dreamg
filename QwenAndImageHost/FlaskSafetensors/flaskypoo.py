import base64
import io
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

# Import necessary libraries for your safetensors model
# Ensure you have these installed: pip install diffusers transformers torch accelerate safetensors
try:
    # IMPORTANT: If your model is an SDXL model, use StableDiffusionXLPipeline.
    # Otherwise, use StableDiffusionPipeline.
    from diffusers import StableDiffusionXLPipeline # Changed to SDXL pipeline
    # from diffusers import StableDiffusionPipeline # Commented out for SDXL
    import torch
    import safetensors
except ImportError:
    print("Warning: Required libraries (diffusers, torch, or safetensors) not found. Model loading will fail.")
    StableDiffusionXLPipeline = None # Changed to SDXL pipeline
    # StableDiffusionPipeline = None # Commented out for SDXL
    torch = None
    safetensors = None

app = Flask(__name__)
CORS(app)

# --- Model Configuration and Loading ---
model_path = "E:/LLM/lustifySDXLNSFW_endgameDMD2/lustifySDXLNSFW_endgameDMD2.safetensors" # <--- IMPORTANT: SET YOUR LOCAL SAFETENSORS FILE PATH HERE

pipe = None
model_loaded = False
load_error_message = ""

def load_model():
    """
    Attempts to load the Stable Diffusion model from the specified path.
    """
    global pipe, model_loaded, load_error_message
    # Check if the correct pipeline class is available (SDXL or regular)
    if StableDiffusionXLPipeline is None or torch is None or safetensors is None: # Changed to SDXL pipeline
        load_error_message = "Required libraries (diffusers, torch, or safetensors) not found, or StableDiffusionXLPipeline is not available. Cannot load model."
        print(load_error_message)
        return

    # --- CUDA Diagnostics ---
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    else:
        print("CUDA is NOT available. Falling back to CPU.")
    # --- End CUDA Diagnostics ---

    if not os.path.exists(model_path):
        load_error_message = f"Model file '{model_path}' does not exist. Please check the path."
        print(load_error_message)
        return

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Attempting to load model '{model_path}' on device: {device}")

        # Use from_single_file for loading a standalone .safetensors checkpoint
        # Using StableDiffusionXLPipeline for potential SDXL model
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32) # Changed to SDXL pipeline
        pipe = pipe.to(device)

        # Optional: If you need to load a VAE explicitly for better quality or if the model
        # expects one and doesn't have it bundled. SDXL models often have a specific VAE.
        # from diffusers import AutoencoderKL
        # pipe.vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

        # Optional: Enable memory-efficient attention (for larger models/limited VRAM)
        # pipe.enable_xformers_memory_efficient_attention() # Requires xformers: pip install xformers

        model_loaded = True
        print(f"Model '{model_path}' loaded successfully on {device}!")
    except Exception as e:
        load_error_message = f"Error loading model '{model_path}': {e}"
        print(load_error_message)
        model_loaded = False

with app.app_context():
    load_model()

def create_placeholder_image(text_content):
    img_width, img_height = 600, 400
    img = Image.new('RGB', (img_width, img_height), color = (255, 99, 71))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    text_color = (255, 255, 255)

    lines = []
    max_chars_per_line = 30
    words = text_content.split(' ')
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars_per_line:
            current_line += (word + " ").strip()
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line.strip())

    y_offset = (img_height - len(lines) * 30) / 2
    for line in lines:
        bbox = d.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (img_width - text_width) / 2
        d.text((x, y_offset), line, fill=text_color, font=font)
        y_offset += 30

    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/generate-image', methods=['POST'])
def generate_image():
    if not request.is_json:
        return jsonify({
            "success": False,
            "job_id": None,
            "image_url": None,
            "error": "Request must be JSON",
            "details": None
        }), 400

    data = request.get_json()
    prompt = data.get('prompt')
    job_id = data.get('job_id') if 'job_id' in data else None

    if not prompt:
        return jsonify({
            "success": False,
            "job_id": job_id,
            "image_url": None,
            "error": "Prompt is required",
            "details": None
        }), 400

    print(f"Received prompt: {prompt}")

    if not model_loaded:
        error_text = f"Model not loaded. {load_error_message}. Using placeholder."
        print(error_text)
        img_str = create_placeholder_image(error_text)
        return jsonify({
            "success": False,
            "job_id": job_id,
            "image_url": f"data:image/png;base64,{img_str}",
            "error": error_text,
            "details": None
        }), 200

    try:
        print(f"Attempting to generate image with prompt: '{prompt}'")
        if pipe is None:
            raise ValueError("Diffusion pipeline (pipe) is None. Model might not have loaded correctly.")

        # --- Clear CUDA cache before generation attempt ---
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")
            print(f"CUDA memory allocated after clear: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            print(f"CUDA memory cached after clear: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        # --- End CUDA cache clear ---

        # Call the pipeline and inspect its output
        # For SDXL, default resolution is often 1024x1024.
        # You might explicitly set height/width if you want a different size or to manage VRAM.
        # Example for SDXL with smaller size (if supported by the model):
        # generation_output = pipe(prompt, height=768, width=768)
        generation_output = pipe(prompt)

        print(f"Type of generation_output: {type(generation_output)}")
        print(f"Attributes of generation_output: {dir(generation_output)}")

        if not hasattr(generation_output, 'images') or not isinstance(generation_output.images, list) or len(generation_output.images) == 0:
            raise ValueError("Model output does not contain 'images' or 'images' is empty/not a list.")

        image = generation_output.images[0]

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "job_id": job_id,
            "image_url": f"data:image/png;base64,{img_str}",
            "error": None,
            "details": None
        }), 200

    except torch.cuda.OutOfMemoryError as e:
        error_message = f"CUDA Out of Memory Error during image generation: {e}. Please try a simpler prompt or free up GPU memory. Using placeholder."
        print(error_message)
        img_str = create_placeholder_image(error_message)
        return jsonify({
            "success": False,
            "job_id": job_id,
            "image_url": f"data:image/png;base64,{img_str}",
            "error": error_message,
            "details": None
        }), 500
    except Exception as e:
        error_message = f"Error during image generation with model: {e}. Using placeholder."
        print(error_message)
        img_str = create_placeholder_image(error_message)
        return jsonify({
            "success": False,
            "job_id": job_id,
            "image_url": f"data:image/png;base64,{img_str}",
            "error": error_message,
            "details": None
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    print("Flask backend running on http://0.0.0.0:5000")
