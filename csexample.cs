// Filename: CSharpApiExample.cs

using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

public class FalApiClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public FalApiClient(string baseUrl = "http://127.0.0.1:5000")
    {
        _httpClient = new HttpClient();
        _baseUrl = baseUrl;
    }

    // Class to represent the JSON payload for the /generate-image endpoint
    public class ImageGenerationRequest
    {
        [JsonProperty("prompt")]
        public string Prompt { get; set; }

        // Important: Use 'lora_url', matching the Flask server's expectation.
        [JsonProperty("lora_url", NullValueHandling = NullValueHandling.Ignore)]
        public string LoraUrl { get; set; }

        [JsonProperty("lora_scale")]
        public double LoraScale { get; set; } = 1.0;
    }

    // Class to represent the JSON response
    public class ImageGenerationResponse
    {
        [JsonProperty("success")]
        public bool Success { get; set; }

        [JsonProperty("job_id")]
        public string JobId { get; set; }

        [JsonProperty("image_url")]
        public string ImageUrl { get; set; }

        [JsonProperty("error")]
        public string Error { get; set; }

        [JsonProperty("details")]
        public string Details { get; set; }
    }

    public async Task<ImageGenerationResponse> GenerateImageAsync(string prompt, string loraUrl = null)
    {
        var requestData = new ImageGenerationRequest
        {
            Prompt = prompt,
            LoraUrl = loraUrl // This will be null if no LoRA is provided
        };

        // Serialize the request object to a JSON string.
        // NullValueHandling.Ignore ensures that if LoraUrl is null, it won't be included in the JSON.
        var jsonPayload = JsonConvert.SerializeObject(requestData, new JsonSerializerSettings
        {
            NullValueHandling = NullValueHandling.Ignore
        });

        var content = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

        try
        {
            // Send the POST request
            var response = await _httpClient.PostAsync($"{_baseUrl}/generate-image", content);
            var responseString = await response.Content.ReadAsStringAsync();

            // Deserialize the JSON response from the server
            var result = JsonConvert.DeserializeObject<ImageGenerationResponse>(responseString);

            if (!response.IsSuccessStatusCode || result == null || !result.Success)
            {
                // Handle errors returned from the API
                var errorMessage = result?.Error ?? "Unknown error";
                var errorDetails = result?.Details ?? responseString;
                Console.WriteLine($"API Error: {errorMessage}\nDetails: {errorDetails}");
                return result;
            }

            return result;
        }
        catch (HttpRequestException e)
        {
            // Handle network errors
            Console.WriteLine($"Request error: {e.Message}");
            return new ImageGenerationResponse { Success = false, Error = e.Message };
        }
    }
}

public class Program
{
    public static async Task Main(string[] args)
    {
        var apiClient = new FalApiClient();

        // --- Example 1: Generate an image without a LoRA ---
        Console.WriteLine("--- Generating image without LoRA ---");
        var response1 = await apiClient.GenerateImageAsync("a beautiful landscape painting");
        if (response1 != null && response1.Success)
        {
            Console.WriteLine($"Success! Job ID: {response1.JobId}");
            Console.WriteLine($"Image URL: {response1.ImageUrl}");
        }
        else
        {
            Console.WriteLine("Image generation failed.");
        }

        Console.WriteLine("\n---------------------------------------\n");

        // --- Example 2: Generate an image WITH a LoRA ---
        // Replace with a real LoRA URL when you have one.
        // The server will fail if this URL is invalid, which is expected.
        string loraUrl = "https://fal.media/files/lora/your_lora.safetensors"; 
        Console.WriteLine("--- Generating image WITH LoRA ---");
        var response2 = await apiClient.GenerateImageAsync("a portrait of a person in the style of the lora", loraUrl);

        if (response2 != null && response2.Success)
        {
            Console.WriteLine($"Success! Job ID: {response2.JobId}");
            Console.WriteLine($"Image URL: {response2.ImageUrl}");
        }
        else
        {
            Console.WriteLine("Image generation with LoRA failed (this might be expected if the LoRA URL is a placeholder).");
        }
    }
} 