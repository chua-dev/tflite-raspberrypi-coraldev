from distutils.command.build_scripts import first_line_re
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image, ImageDraw, ImageFont
import pdb

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/face_detection_full_range.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
_, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
print(f'{input_height} height + {input_width} width')

output_details = interpreter.get_output_details()
print(output_details)

# Load the image
test_image = cv2.imread('photo/noface1.jpeg')

# IMAGE METHOD 1
#test_image = Image.fromarray(test_image)
#print(test_image)
#raw_image = test_image.resize((input_width, input_height), Image.ANTIALIAS)
#print(raw_image)

# IMAGE METHOD 2
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
test_image = cv2.resize(test_image, (input_width, input_height))
print(test_image)
raw_image = np.expand_dims(test_image, axis=0)
print(raw_image)

input_mean = 127.5
input_std = 127.5

#raw_image = (np.float32(raw_image) - input_mean) / input_std
raw_image = np.float32(raw_image)

# Putting image for tflite model to predict & Invoke
interpreter.set_tensor(input_details[0]['index'], raw_image)
interpreter.invoke()

# Getting Output & Result
boxes = interpreter.get_tensor(output_details[0]['index'])[0]
classes = interpreter.get_tensor(output_details[1]['index'])[0]
print(boxes)
print(classes)

# TESTING

pdb.set_trace()
