# -*- coding: utf-8 -*-
"""
visulize transformed images of the first layer
@author: yz
"""
import argparse
import caffe
import numpy as np
import cv2
from google.protobuf import text_format

# parse input
parser = argparse.ArgumentParser(
    description = 'Plot transformed input images')
parser.add_argument('--MODEL_DEF', type = str,
                    default = "")
parser.add_argument('--MODEL_WEIGHTS', type = str,
                    default = None)
parser.add_argument('--REPEAT', type = int, default = 10)
parser.add_argument('--PHASE', type = str, default = 'TRAIN')
parser.add_argument('--DATA_LAYER_NAME', type  = str, default = None)

# reassian input
parsed = parser.parse_args()
model_def = parsed.MODEL_DEF
model_weights = parsed.MODEL_WEIGHTS
repeat = parsed.REPEAT
phase = parsed.PHASE.lower()
layername = parsed.DATA_LAYER_NAME

# load model
caffe.set_mode_cpu()
net = None
caffephase = getattr(caffe, phase.upper())
if model_weights:
    net = caffe.Net(model_def, model_weights, caffephase)
else:
    net = caffe.Net(model_def, caffephase)

# get the first blobs
if not layername:
    layername = net.blobs.keys()[0]

# get net parameters
netparam = caffe.proto.caffe_pb2.NetParameter()
with open(model_def, 'r') as model:
    text_format.Merge(str(model.read()), netparam)
# find eps value
meanvalue = 0
for layer in netparam.layer:
    if layer.name == layername:
        meanvalue = layer.transform_param.mean_value

# get first layer output data
index = 0
while index < repeat:
    net.forward(end = layername)
    inputs = net.blobs[layername].data
    for i in range(inputs.shape[0]):
        img = np.transpose(inputs[i,:,:,:]+meanvalue,
                           (1, 2, 0)).astype(np.uint8)
        filename = str(i)
        cv2.namedWindow(filename)
        cv2.moveWindow(filename, 10, 50)
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        for iwait in range(1, 5): cv2.waitKey(1)
        index += 1
        if index >= repeat:
            break