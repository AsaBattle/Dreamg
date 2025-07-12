# AI Image Generation API

A Flask-based API server for generating images and creating LoRAs using the Fal.ai platform with the CyberRealistic Pony SDXL model. Designed to integrate with C# applications for interactive story generation with consistent character illustrations.

## 🚀 Features

- **Image Generation**: Generate high-quality images from text prompts using the CyberRealistic Pony SDXL model
- **Custom LoRA Support**: Create and use custom LoRAs for consistent character generation
- **Job Tracking**: Monitor the progress of image generation and LoRA training
- **REST API**: Easy integration with any application via HTTP requests
- **C# Client Library**: Complete C# client code included for easy integration
- **Base64 Support**: Return images as base64 for direct embedding in applications

## 📋 Requirements

- Python 3.8+
- Fal.ai API key
- Windows, macOS, or Linux

## 🛠️ Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create Virtual Environment
```bash
# Windows
py -m venv venv
.\venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Edit `flask_server.py` and replace the API key with your own Fal.ai API key:
```python
os.environ["FAL_KEY"] = "your_fal_ai_api_key_here"
```

### 5. Start the Server
```bash
python flask_server.py
```

The server will start on `http://localhost:5000`

## 🌐 API Endpoints

### Health Check
```
GET /health
```

### Generate Image
```
POST /generate-image
Content-Type: application/json

{
    "prompt": "a cyberpunk warrior with neon armor",
    "lora_url": "optional_lora_url",
    "lora_scale": 1.0,
    "return_base64": false
}
```

### Create LoRA
```
POST /create-lora
Content-Type: application/json

{
    "name": "character_name",
    "images": ["base64_image1", "base64_image2", ...],
    "steps": 1000
}
```

### Job Status
```
GET /job-status/{job_id}
```

### List LoRAs
```
GET /list-loras
```

### Download Image
```
GET /download-image/{job_id}
```

## 🧪 Testing

Run the test script to verify everything is working:
```bash
python test_api.py
```

## 🔧 C# Integration

For C# projects, use the included `CSharpApiExample.cs` file. Add these NuGet packages:
- `Newtonsoft.Json`
- `System.Net.Http`

Example usage:
```csharp
var client = new ImageGenerationClient();

// Generate an image
var result = await client.GenerateImageAsync(
    "a medieval knight in a dark forest"
);

// Use with a custom LoRA
var resultWithLora = await client.GenerateImageAsync(
    "the main character walking through the village",
    loraUrl: "your_lora_url_here"
);
```

## 📁 Project Structure

```
├── flask_server.py          # Main Flask API server
├── generate_image.py        # Standalone image generation script
├── create_lora_simple.py    # Standalone LoRA creation script
├── test_api.py             # API testing script
├── CSharpApiExample.cs     # C# client library
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## 🎨 Model Information

This project uses the **CyberRealistic Pony v8 SDXL** model, which is specifically designed for:
- Photorealistic image generation
- High-quality character illustrations
- Excellent prompt following
- LoRA compatibility

The model automatically applies Pony Diffusion formatting to prompts for optimal results.

## 🔒 Security Notes

- Keep your Fal.ai API key secure and never commit it to version control
- Consider using environment variables for API keys in production
- The server runs in debug mode by default - disable for production use

## 📝 License

This project is for educational and personal use. Please respect the licenses of:
- Fal.ai service terms
- CyberRealistic Pony model license
- Any LoRAs you create or use

## 🤝 Contributing

Feel free to submit issues and pull requests to improve the project.

## 📞 Support

If you encounter any issues:
1. Check the Flask server logs
2. Verify your Fal.ai API key is valid
3. Ensure all dependencies are installed correctly
4. Test with the included `test_api.py` script 