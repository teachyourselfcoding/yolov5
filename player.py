import cv2
import numpy as np

# read the ImageNet class names
import onnx
onnx_model = onnx.load( 'model.onnx')
onnx.checker.check_model(onnx_model)