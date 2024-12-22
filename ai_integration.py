import os
from PIL import Image
import io
import base64
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import logging

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def get_gemini_model():
    return GoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7
    )

def analyze_depth_image(real_image: Image.Image, depth_image: Image.Image) -> str:
    try:
        # Resize the depth image to reduce its size before converting to base64 (optional step)
        max_size = 1024  # Set a reasonable max size for images
        if max(depth_image.size) > max_size:
            depth_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logging.info(f"Depth image resized to {depth_image.size} for efficient processing.")

        # Convert depth image to base64
        img_byte_arr = io.BytesIO()
        depth_image.save(img_byte_arr, format='PNG', optimize=True, compress_level=9)  # Using optimization for PNG
        base64_depth_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Resize real image for potential analysis, ensuring it's not too large
        if max(real_image.size) > max_size:
            real_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            logging.info(f"Real image resized to {real_image.size} for efficient processing.")

        img_byte_arr_real = io.BytesIO()
        real_image.save(img_byte_arr_real, format='PNG', optimize=True, compress_level=9)
        base64_real_image = base64.b64encode(img_byte_arr_real.getvalue()).decode('utf-8')

        # Load the generative AI model
        model = get_gemini_model()

        # Define the input prompt
        system_message = SystemMessage(content=""" 
            You are an expert in spatial analysis using depth maps. 
            Darker regions indicate objects farther away, and lighter regions represent closer objects. 
            Analyze the image and provide a one-line situational description focusing on:
            1. Open spaces and their locations.
            2. Obstacles or blocked areas.
            3. Suggested navigation paths for a blind individual.
        """)
        human_message = HumanMessage(content=[ 
            {"type": "image", "image": base64_depth_image},
            {"type": "image", "image": base64_real_image},
            {"type": "text", "text": "Provide a one-line analysis for blind navigation using both depth and real images."}
        ])

        # Get response from the AI model
        response = model.invoke([system_message, human_message])
        return response.strip() if response else "No meaningful insights generated."
    
    except Exception as e:
        logging.error(f"Error analyzing depth and real images: {str(e)}")
        return f"Error analyzing depth and real images: {str(e)}"
