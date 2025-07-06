// CSharpApiExample.cs
// Example C# code for calling the Flask Image Generation API
// Add these NuGet packages to your project:
// - Newtonsoft.Json
// - System.Net.Http

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;

namespace ImageGenerationApi
{
    public class ImageGenerationClient
    {
        private readonly HttpClient _httpClient;
        private readonly string _baseUrl;

        public ImageGenerationClient(string baseUrl = "http://localhost:5000")
        {
            _httpClient = new HttpClient();
            _baseUrl = baseUrl;
        }

        #region Data Models

        public class HealthResponse
        {
            public string Status { get; set; }
            public string Timestamp { get; set; }
            public string Version { get; set; }
        }

        public class ImageGenerationRequest
        {
            [JsonProperty("prompt")]
            public string Prompt { get; set; }

            [JsonProperty("lora_url")]
            public string LoraUrl { get; set; }

            [JsonProperty("lora_scale")]
            public double LoraScale { get; set; } = 1.0;

            [JsonProperty("return_base64")]
            public bool ReturnBase64 { get; set; } = false;
        }

        public class ImageGenerationResponse
        {
            [JsonProperty("success")]
            public bool Success { get; set; }

            [JsonProperty("job_id")]
            public string JobId { get; set; }

            [JsonProperty("image_url")]
            public string ImageUrl { get; set; }

            [JsonProperty("prompt_used")]
            public string PromptUsed { get; set; }

            [JsonProperty("image_base64")]
            public string ImageBase64 { get; set; }

            [JsonProperty("error")]
            public string Error { get; set; }
        }

        public class LoraCreationRequest
        {
            [JsonProperty("name")]
            public string Name { get; set; }

            [JsonProperty("images")]
            public List<string> Images { get; set; }

            [JsonProperty("steps")]
            public int Steps { get; set; } = 1000;
        }

        public class LoraCreationResponse
        {
            [JsonProperty("success")]
            public bool Success { get; set; }

            [JsonProperty("job_id")]
            public string JobId { get; set; }

            [JsonProperty("lora_name")]
            public string LoraName { get; set; }

            [JsonProperty("lora_url")]
            public string LoraUrl { get; set; }

            [JsonProperty("config_url")]
            public string ConfigUrl { get; set; }

            [JsonProperty("error")]
            public string Error { get; set; }
        }

        public class JobStatusResponse
        {
            [JsonProperty("success")]
            public bool Success { get; set; }

            [JsonProperty("job_id")]
            public string JobId { get; set; }

            [JsonProperty("job_info")]
            public JobInfo JobInfo { get; set; }

            [JsonProperty("error")]
            public string Error { get; set; }
        }

        public class JobInfo
        {
            [JsonProperty("type")]
            public string Type { get; set; }

            [JsonProperty("status")]
            public string Status { get; set; }

            [JsonProperty("logs")]
            public List<string> Logs { get; set; }

            [JsonProperty("created")]
            public string Created { get; set; }

            [JsonProperty("last_update")]
            public string LastUpdate { get; set; }

            [JsonProperty("image_url")]
            public string ImageUrl { get; set; }

            [JsonProperty("lora_url")]
            public string LoraUrl { get; set; }

            [JsonProperty("error")]
            public string Error { get; set; }
        }

        public class LoraInfo
        {
            [JsonProperty("name")]
            public string Name { get; set; }

            [JsonProperty("created")]
            public string Created { get; set; }

            [JsonProperty("lora_url")]
            public string LoraUrl { get; set; }

            [JsonProperty("config_url")]
            public string ConfigUrl { get; set; }

            [JsonProperty("steps")]
            public int Steps { get; set; }

            [JsonProperty("job_id")]
            public string JobId { get; set; }
        }

        public class LoraListResponse
        {
            [JsonProperty("success")]
            public bool Success { get; set; }

            [JsonProperty("loras")]
            public List<LoraInfo> Loras { get; set; }

            [JsonProperty("error")]
            public string Error { get; set; }
        }

        #endregion

        #region API Methods

        /// <summary>
        /// Check if the API server is healthy and responding
        /// </summary>
        public async Task<HealthResponse> CheckHealthAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/health");
                var content = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<HealthResponse>(content);
            }
            catch (Exception ex)
            {
                throw new Exception($"Health check failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Generate an image from a text prompt
        /// </summary>
        /// <param name="prompt">Text description of the image to generate</param>
        /// <param name="loraUrl">Optional LoRA URL to apply</param>
        /// <param name="loraScale">LoRA strength (0.0 to 1.0)</param>
        /// <param name="returnBase64">Whether to return the image as base64 data</param>
        /// <returns>Image generation response</returns>
        public async Task<ImageGenerationResponse> GenerateImageAsync(
            string prompt, 
            string loraUrl = null, 
            double loraScale = 1.0, 
            bool returnBase64 = false)
        {
            try
            {
                var request = new ImageGenerationRequest
                {
                    Prompt = prompt,
                    LoraUrl = loraUrl,
                    LoraScale = loraScale,
                    ReturnBase64 = returnBase64
                };

                var json = JsonConvert.SerializeObject(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}/generate-image", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    return JsonConvert.DeserializeObject<ImageGenerationResponse>(responseContent);
                }
                else
                {
                    var errorResponse = JsonConvert.DeserializeObject<ImageGenerationResponse>(responseContent);
                    throw new Exception($"Image generation failed: {errorResponse.Error}");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Image generation request failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Create a LoRA from a collection of images
        /// </summary>
        /// <param name="name">Name for the LoRA</param>
        /// <param name="imageFiles">List of image file paths</param>
        /// <param name="steps">Number of training steps</param>
        /// <returns>LoRA creation response</returns>
        public async Task<LoraCreationResponse> CreateLoraAsync(
            string name, 
            List<string> imageFiles, 
            int steps = 1000)
        {
            try
            {
                // Convert image files to base64
                var base64Images = new List<string>();
                foreach (var filePath in imageFiles)
                {
                    if (!File.Exists(filePath))
                    {
                        throw new FileNotFoundException($"Image file not found: {filePath}");
                    }

                    var imageBytes = await File.ReadAllBytesAsync(filePath);
                    var base64 = Convert.ToBase64String(imageBytes);
                    base64Images.Add(base64);
                }

                var request = new LoraCreationRequest
                {
                    Name = name,
                    Images = base64Images,
                    Steps = steps
                };

                var json = JsonConvert.SerializeObject(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_baseUrl}/create-lora", content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    return JsonConvert.DeserializeObject<LoraCreationResponse>(responseContent);
                }
                else
                {
                    var errorResponse = JsonConvert.DeserializeObject<LoraCreationResponse>(responseContent);
                    throw new Exception($"LoRA creation failed: {errorResponse.Error}");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"LoRA creation request failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Get the status of a job (image generation or LoRA creation)
        /// </summary>
        /// <param name="jobId">Job ID to check</param>
        /// <returns>Job status response</returns>
        public async Task<JobStatusResponse> GetJobStatusAsync(string jobId)
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/job-status/{jobId}");
                var content = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    return JsonConvert.DeserializeObject<JobStatusResponse>(content);
                }
                else
                {
                    var errorResponse = JsonConvert.DeserializeObject<JobStatusResponse>(content);
                    throw new Exception($"Job status check failed: {errorResponse.Error}");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Job status request failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Get a list of all created LoRAs
        /// </summary>
        /// <returns>List of LoRAs</returns>
        public async Task<LoraListResponse> GetLorasAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/list-loras");
                var content = await response.Content.ReadAsStringAsync();
                return JsonConvert.DeserializeObject<LoraListResponse>(content);
            }
            catch (Exception ex)
            {
                throw new Exception($"LoRA list request failed: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Download an image from a job
        /// </summary>
        /// <param name="jobId">Job ID</param>
        /// <param name="savePath">Local path to save the image</param>
        public async Task DownloadImageAsync(string jobId, string savePath)
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_baseUrl}/download-image/{jobId}");
                
                if (response.IsSuccessStatusCode)
                {
                    var imageBytes = await response.Content.ReadAsByteArrayAsync();
                    await File.WriteAllBytesAsync(savePath, imageBytes);
                }
                else
                {
                    throw new Exception($"Image download failed with status: {response.StatusCode}");
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Image download failed: {ex.Message}", ex);
            }
        }

        #endregion

        public void Dispose()
        {
            _httpClient?.Dispose();
        }
    }

    #region Example Usage

    public class Program
    {
        public static async Task Main(string[] args)
        {
            var client = new ImageGenerationClient();

            try
            {
                // Check server health
                Console.WriteLine("🔍 Checking server health...");
                var health = await client.CheckHealthAsync();
                Console.WriteLine($"Server status: {health.Status}");
                Console.WriteLine();

                // Generate an image
                Console.WriteLine("🎨 Generating image...");
                var imageResult = await client.GenerateImageAsync(
                    "a cyberpunk warrior with neon armor standing in a futuristic city"
                );

                if (imageResult.Success)
                {
                    Console.WriteLine($"✅ Image generated successfully!");
                    Console.WriteLine($"Job ID: {imageResult.JobId}");
                    Console.WriteLine($"Image URL: {imageResult.ImageUrl}");

                    // Check job status
                    Console.WriteLine("\n📊 Checking job status...");
                    var status = await client.GetJobStatusAsync(imageResult.JobId);
                    Console.WriteLine($"Job status: {status.JobInfo.Status}");
                }
                else
                {
                    Console.WriteLine($"❌ Image generation failed: {imageResult.Error}");
                }

                // List available LoRAs
                Console.WriteLine("\n📋 Listing available LoRAs...");
                var loras = await client.GetLorasAsync();
                Console.WriteLine($"Found {loras.Loras.Count} LoRAs");

                foreach (var lora in loras.Loras)
                {
                    Console.WriteLine($"  - {lora.Name} (created: {lora.Created})");
                }

                // Example of creating a LoRA (commented out as it requires actual image files)
                /*
                Console.WriteLine("\n🚀 Creating LoRA...");
                var imageFiles = new List<string>
                {
                    @"C:\path\to\image1.jpg",
                    @"C:\path\to\image2.jpg",
                    @"C:\path\to\image3.jpg",
                    @"C:\path\to\image4.jpg",
                    @"C:\path\to\image5.jpg"
                };

                var loraResult = await client.CreateLoraAsync("my_character", imageFiles);
                if (loraResult.Success)
                {
                    Console.WriteLine($"✅ LoRA created: {loraResult.LoraUrl}");
                }
                */

            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Error: {ex.Message}");
            }
            finally
            {
                client.Dispose();
            }
        }
    }

    #endregion
} 