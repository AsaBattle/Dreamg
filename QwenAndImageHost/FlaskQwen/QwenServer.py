#!/usr/bin/env python3
"""
Qwen3-8B Local Server
A Flask-based server for running Qwen3-8B model locally on localhost:5000
Supports both thinking and non-thinking modes with OpenAI-compatible API

Required packages:
pip install torch transformers flask accelerate bitsandbytes
"""

import json
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS # Import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
import threading
from typing import Dict, List, Optional, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class Qwen3Server:
    def __init__(self, model_name: str = r"E:\LLM\Qwen3-8B"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.load_lock = threading.Lock()
        self.device = self._get_device()
        self.device = self._get_device()
        
    def _get_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return device
        else:
            logger.info("CUDA not available, using CPU")
            return torch.device("cpu")
        
    def load_model(self):
        """Load the Qwen3-8B model and tokenizer"""
        if self.model_loaded:
            return
            
        with self.load_lock:
            if self.model_loaded:
                return
                
            logger.info(f"Loading model from local path: {self.model_name}")
            
            # Check if local model path exists
            import os
            if not os.path.exists(self.model_name):
                logger.error(f"Model path does not exist: {self.model_name}")
                raise FileNotFoundError(f"Model path not found: {self.model_name}")
            
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                # Configure model loading based on available hardware
                model_kwargs = {
                    "trust_remote_code": True,
                    "local_files_only": True
                }
                
                if self.device.type == "cuda":
                    # Use CUDA optimizations
                    model_kwargs.update({
                        "torch_dtype": torch.float16,  # Use half precision for better memory usage
                        "device_map": "auto",  # Automatically distribute model across available GPUs
                        "low_cpu_mem_usage": True
                    })
                    
                    # Check if we should use quantization for 8GB GPU
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    if gpu_memory_gb <= 10:  # Use 4-bit quantization for GPUs with 10GB or less
                        logger.info("Using 4-bit quantization for better memory efficiency")
                        model_kwargs.update({
                            "load_in_4bit": True,
                            "bnb_4bit_compute_dtype": torch.float16,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True
                        })
                else:
                    # CPU configuration
                    model_kwargs.update({
                        "torch_dtype": torch.float32
                    })
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
                
                # Move to device if not using device_map
                if "device_map" not in model_kwargs:
                    self.model = self.model.to(self.device)
                
                self.model_loaded = True
                logger.info("Model loaded successfully from local path!")
                
                # Print memory usage if CUDA
                if self.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"GPU Memory - Allocated: {memory_allocated:.1f} GB, Reserved: {memory_reserved:.1f} GB")
                    
            except Exception as e:
                logger.error(f"Failed to load model from local path: {e}")
                raise
    
    def generate_response(self, messages: List[Dict], enable_thinking: bool = True,
                          temperature: float = None, top_p: float = None,
                          top_k: int = None, max_tokens: int = 32768) -> Dict:
        """Generate response from the model"""
        if not self.model_loaded:
            self.load_model()

        # Set default parameters based on thinking mode
        if enable_thinking:
            temperature = temperature or 0.6
            top_p = top_p or 0.95
            top_k = top_k or 20
        else:
            temperature = temperature or 0.7
            top_p = top_p or 0.8
            top_k = top_k or 20

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        # Tokenize input
        model_inputs = self.tokenizer([text], return_tensors="pt")

        # Move inputs to the correct device (Apply same logic for both CUDA and CPU)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        output_ids = generated_ids[0][len(model_inputs["input_ids"][0]):].tolist() # Access via dictionary key

        # Parse thinking content if in thinking mode
        thinking_content = ""
        content = ""

        if enable_thinking:
            try:
                # Find </think> token (151668)
                # Note: This index approach assumes the model will reliably generate the closing tag.
                # If the model sometimes omits it, this will raise a ValueError.
                # A more robust approach might involve token-level parsing or looking for specific string patterns.
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        else:
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        return {
            "thinking_content": thinking_content,
            "content": content,
            "enable_thinking": enable_thinking
        }
    
    def generate_stream(self, messages: List[Dict], enable_thinking: bool = True,
                       temperature: float = None, top_p: float = None,
                       top_k: int = None, max_tokens: int = 32768) -> Generator[str, None, None]:
        """Generate streaming response (simplified version)"""
        result = self.generate_response(messages, enable_thinking, temperature, top_p, top_k, max_tokens)
        
        # Simulate streaming by yielding chunks
        full_response = result["content"]
        if enable_thinking and result["thinking_content"]:
            full_response = f"<think>{result['thinking_content']}</think>\n{result['content']}"
        
        words = full_response.split()
        for i, word in enumerate(words):
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + " " if i < len(words) - 1 else word},
                    "finish_reason": None if i < len(words) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.05)  # Simulate streaming delay
        
        yield "data: [DONE]\n\n"

# Initialize the server
qwen_server = Qwen3Server()

@app.route('/health', methods=['GET'])
def health_check():
    request_time = time.time()
    logger.info(f"[health_check] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """Health check endpoint"""
    device_info = {
        "device": str(qwen_server.device),
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        })
        
        if qwen_server.model_loaded:
            device_info.update({
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.1f} GB",
                "gpu_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.1f} GB"
            })
    
    response = jsonify({
        "status": "healthy", 
        "model_loaded": qwen_server.model_loaded,
        "device_info": device_info
    })
    resolved_time = time.time()
    logger.info(f"[health_check] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
    return response

@app.route('/v1/models', methods=['GET'])
def list_models():
    request_time = time.time()
    logger.info(f"[list_models] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """OpenAI-compatible models endpoint"""
    response = jsonify({
        "object": "list",
        "data": [{
            "id": qwen_server.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "qwen"
        }]
    })
    resolved_time = time.time()
    logger.info(f"[list_models] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
    return response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    request_time = time.time()
    logger.info(f"[chat_completions] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.get_json()
        
        # Extract parameters
        messages = data.get('messages', [])
        enable_thinking = data.get('enable_thinking', True)
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        max_tokens = data.get('max_tokens', 32768)
        stream = data.get('stream', False)
        
        # Extract top_k from extra parameters if provided
        top_k = data.get('top_k')
        
        if not messages:
            return jsonify({"error": "Messages are required"}), 400
        
        if stream:
            # Streaming response
            response = Response(
                qwen_server.generate_stream(messages, enable_thinking, temperature, top_p, top_k, max_tokens),
                mimetype='text/plain'
            )
        else:
            # Non-streaming response
            result = qwen_server.generate_response(messages, enable_thinking, temperature, top_p, top_k, max_tokens)
            response = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": qwen_server.model_name,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["content"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # Would need to implement token counting
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            # Add thinking content if available
            if enable_thinking and result["thinking_content"]:
                response["thinking_content"] = result["thinking_content"]
            response = jsonify(response)
        resolved_time = time.time()
        logger.info(f"[chat_completions] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
        return response
            
    except Exception as e:
        logger.error(f"Error in chat completions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    request_time = time.time()
    logger.info(f"[generate] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """Simple generate endpoint"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        enable_thinking = data.get('enable_thinking', False)
        temperature = data.get('temperature')
        top_p = data.get('top_p')
        top_k = data.get('top_k')
        max_tokens = data.get('max_tokens', 32768)
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        messages = [{"role": "user", "content": prompt}]
        logger.info(f"Generating response for prompt: {prompt[:10]} (Thinking mode: {enable_thinking})")
        result = qwen_server.generate_response(messages, enable_thinking, temperature, top_p, top_k, max_tokens)
        
        response = jsonify(result)
        resolved_time = time.time()
        logger.info(f"[generate] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
        return response
        
    except Exception as e:
        logger.error(f"Error in generate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/thinking_mode', methods=['POST'])
def toggle_thinking_mode():
    request_time = time.time()
    logger.info(f"[toggle_thinking_mode] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """Endpoint to test thinking mode switching"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', 'Explain quantum computing.')
        
        # Test both modes
        messages = [{"role": "user", "content": prompt}]
        
        thinking_result = qwen_server.generate_response(messages, enable_thinking=True)
        non_thinking_result = qwen_server.generate_response(messages, enable_thinking=False)
        
        response = jsonify({
            "thinking_mode": {
                "thinking_content": thinking_result["thinking_content"],
                "content": thinking_result["content"]
            },
            "non_thinking_mode": {
                "content": non_thinking_result["content"]
            }
        })
        resolved_time = time.time()
        logger.info(f"[toggle_thinking_mode] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
        return response
        
    except Exception as e:
        logger.error(f"Error in thinking mode test: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    request_time = time.time()
    logger.info(f"[index] Requested at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_time))}")
    """Root endpoint with usage information"""
    response = jsonify({
        "message": "Qwen3-8B Local Server",
        "model": qwen_server.model_name,
        "endpoints": {
            "/health": "Health check",
            "/v1/models": "List available models",
            "/v1/chat/completions": "OpenAI-compatible chat completions",
            "/generate": "Simple text generation",
            "/thinking_mode": "Test thinking mode switching"
        },
        "example_usage": {
            "curl_example": 'curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d "{\"prompt\": \"Write a short story about AI\", \"enable_thinking\": true}"',
            "openai_compatible": 'curl -X POST http://localhost:5000/v1/chat/completions -H "Content-Type: application/json" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}], \"enable_thinking\": true}"'
        }
    })
    resolved_time = time.time()
    logger.info(f"[index] Resolved at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(resolved_time))}")
    return response

if __name__ == '__main__':
    print("Starting Qwen3-8B Local Server...")
    print(f"Model path: {qwen_server.model_name}")
    print(f"Device: {qwen_server.device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print("Loading model (this may take a few minutes)...")
    
    # Pre-load the model
    try:
        qwen_server.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Server will attempt to load model on first request")
        print("Make sure you have installed: pip install torch transformers flask accelerate bitsandbytes")
        print("And ensure the model files exist at: E:\\LLM\\Qwen3-8B")
    
    print("Server starting on http://localhost:5000")
    print("Visit http://localhost:5000 for usage information")
    
    app.run(port=5000, debug=True, threaded=True)