# Import the library
import google.generativeai as genai
import os

# Configure the API key
# You'll need to get an API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY = "AIzaSyDioCk3CkTGyniXVYGY6cP3d_cpWtIzYdk"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the model
model = genai.GenerativeModel('gemini-2.0-flash')

# Example 1: Simple text generation
def generate_text(prompt):
    """Generate text using Gemini"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example 2: Chat conversation
def chat_with_gemini():
    """Have a conversation with Gemini"""
    chat = model.start_chat(history=[])
    
    # Example conversation
    response = chat.send_message("Hello! Can you explain what machine learning is?")
    print("Gemini:", response.text)
    
    response = chat.send_message("What are the main types of machine learning?")
    print("Gemini:", response.text)
    
    return chat

# Example 3: Generate content with specific parameters
def generate_with_params(prompt, temperature=0.7, max_tokens=1000):
    """Generate content with custom parameters"""
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example 4: Stream responses
def stream_response(prompt):
    """Stream the response from Gemini"""
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            print(chunk.text, end='', flush=True)
        print()  # New line at the end
    except Exception as e:
        print(f"Error: {str(e)}")

# Example 5: Multi-modal (text + image) - requires gemini-pro-vision
def analyze_image(image_path, prompt="Describe this image"):
    """Analyze an image using Gemini Vision"""
    try:
        # Note: This requires gemini-pro-vision model
        vision_model = genai.GenerativeModel('gemini-pro-vision')
        
        # Load image
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
        
        response = vision_model.generate_content([prompt, img_data])
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Test the functions
if __name__ == "__main__":
    # Test simple generation
    print("=== Simple Text Generation ===")
    result = generate_text("Write a short poem about technology")
    print(result)
    print("\n")
    
    # Test chat
    print("=== Chat with Gemini ===")
    chat_with_gemini()
    print("\n")
    
    # Test with parameters
    print("=== Generation with Parameters ===")
    result = generate_with_params("Explain quantum computing in simple terms", temperature=0.3)
    print(result)
    print("\n")
    
    # Test streaming
    print("=== Streaming Response ===")
    stream_response("Write a creative story about a robot learning to paint")
