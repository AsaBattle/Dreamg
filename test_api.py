# test_api.py
# Simple test script to demonstrate API usage (reference for C# implementation)
# here is a brand new updated version of the api
import requests
import json
import time
import base64

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_image_generation():
    """Test image generation endpoint"""
    print("🎨 Testing image generation...")
    
    # Request payload
    payload = {
        "prompt": "a cyberpunk warrior with neon armor",
        "return_base64": False  # Set to True if you want base64 encoded image
    }
    
    response = requests.post(f"{BASE_URL}/generate-image", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        print(f"Job ID: {result['job_id']}")
        print(f"Image URL: {result['image_url']}")
        return result['job_id']
    else:
        print(f"Error: {response.json()}")
        return None
    print()

def test_job_status(job_id):
    """Test job status endpoint"""
    if not job_id:
        return
    
    print(f"📊 Testing job status for {job_id}...")
    response = requests.get(f"{BASE_URL}/job-status/{job_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_list_loras():
    """Test listing LoRAs"""
    print("📋 Testing LoRA listing...")
    response = requests.get(f"{BASE_URL}/list-loras")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_lora_creation():
    """Test LoRA creation (with dummy base64 images)"""
    print("🚀 Testing LoRA creation...")
    
    # Create dummy base64 images (in real use, these would be actual image data)
    # For testing, we'll create small 1x1 pixel images
    dummy_images = []
    for i in range(5):  # Need at least 5 images
        # Create a simple 1x1 red pixel PNG in base64
        # This is just for testing - real images should be proper photos
        dummy_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        dummy_images.append(dummy_base64)
    
    payload = {
        "name": "test_lora",
        "images": dummy_images,
        "steps": 100  # Use fewer steps for testing
    }
    
    print("⚠️  Note: This will fail because we're using dummy 1x1 pixel images.")
    print("In real use, you would provide actual base64-encoded photos.")
    
    response = requests.post(f"{BASE_URL}/create-lora", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

if __name__ == "__main__":
    print("🧪 API Test Script")
    print("Make sure the Flask server is running first!")
    print("=" * 50)
    
    try:
        # Test health check
        test_health_check()
        
        # Test image generation
        job_id = test_image_generation()
        
        # Test job status
        if job_id:
            time.sleep(2)  # Wait a bit for job to process
            test_job_status(job_id)
        
        # Test listing LoRAs
        test_list_loras()
        
        # Test LoRA creation (will fail with dummy images)
        test_lora_creation()
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server.")
        print("Make sure to run: python flask_server.py")
    except Exception as e:
        print(f"❌ Error: {e}") 