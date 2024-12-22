import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import tensorflow as tf
from ai_integration import analyze_depth_image
import base64
import time  # For measuring time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all, 1 = warnings, 2 = errors, 3 = none
import tensorflow as tf




app = Flask(__name__)

MODEL_PATH = 'Midas-V2.tflite'

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_HEIGHT = 256
INPUT_WIDTH = 256
INPUT_MEAN = 127.5
INPUT_STD = 127.5

MAX_IMAGE_SIZE = 65535  # Maximum allowed size for the image in bytes

def preprocess_image(image: Image.Image) -> tuple:
    start_time = time.time()  # Start timing

    original_size = image.size

    # Resize the image to 256x256 for input processing
    image = image.resize((INPUT_WIDTH, INPUT_HEIGHT))
    
    # Convert image to byte array to check its size
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_size_in_bytes = len(img_byte_arr.getvalue())

    if img_size_in_bytes > MAX_IMAGE_SIZE:
        # If the image is larger than the max size, resize it
        scale_factor = (MAX_IMAGE_SIZE / img_size_in_bytes) ** 0.5
        new_width = int(image.width * scale_factor)
        new_height = int(image.height * scale_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"Resized image to {new_width}x{new_height} to fit within the size limit.")

    image = image.convert('RGB')
    image_np = np.array(image).astype(np.float32)
    image_np = (image_np - INPUT_MEAN) / INPUT_STD
    image_np = np.expand_dims(image_np, axis=0)

    end_time = time.time()  # End timing
    print(f"Preprocessing time: {end_time - start_time:.4f} seconds")
    return image_np, original_size

def postprocess_depth(depth: np.ndarray) -> Image.Image:
    start_time = time.time()  # Start timing

    depth = np.squeeze(depth)
    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    depth_image = (depth_normalized * 255).astype(np.uint8)
    depth_pil = Image.fromarray(depth_image)

    # Resize depth image to 256x256
    depth_pil = depth_pil.resize((INPUT_WIDTH, INPUT_HEIGHT), Image.Resampling.LANCZOS)

    end_time = time.time()  # End timing
    print(f"Postprocessing time: {end_time - start_time:.4f} seconds")
    return depth_pil

@app.route('/depth', methods=['POST'])
def generate_depth():
    overall_start = time.time()  # Overall start time

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    try:
        step_start = time.time()
        real_image = Image.open(file.stream)
        print(f"Image loading time: {time.time() - step_start:.4f} seconds")

        step_start = time.time()
        input_data, original_size = preprocess_image(real_image)
        print(f"Preprocessing total time: {time.time() - step_start:.4f} seconds")

        step_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        print(f"Model inference time: {time.time() - step_start:.4f} seconds")

        step_start = time.time()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        depth = output_data[0]
        print(f"Depth extraction time: {time.time() - step_start:.4f} seconds")

        step_start = time.time()
        depth_image = postprocess_depth(depth)
        print(f"Postprocessing total time: {time.time() - step_start:.4f} seconds")

        # Now call analyze_depth_image with both real_image and depth_image
        step_start = time.time()
        analysis = analyze_depth_image(real_image, depth_image)  # Pass both real and depth images
        print(f"Depth and real image analysis time: {time.time() - step_start:.4f} seconds")

        overall_end = time.time()
        print(f"Overall request handling time: {overall_end - overall_start:.4f} seconds")

        # Create a response
        img_io = io.BytesIO()
        depth_image = depth_image.convert('L')
        depth_image.save(img_io, 'PNG', optimize=True, compress_level=9)
        img_io.seek(0)

        if request.headers.get('Accept') == 'application/json':
            img_io.seek(0)
            img_base64 = base64.b64encode(img_io.getvalue()).decode()
            return jsonify({
                'depth_map': img_base64,
                'analysis': analysis
            })
        else:
            img_io.seek(0)
            return send_file(img_io, mimetype='image/png')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Depth Map Generator</title>
    <h1>Upload an image to get its depth map</h1>
    <form method=post enctype=multipart/form-data action="/depth">
      <input type=file name=image>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
