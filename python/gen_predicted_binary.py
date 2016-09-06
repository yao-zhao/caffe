# generate predicted probability for binary classes
# output probability for class 0 and 1
# output: first column image file, second column label,
# third column probability for class 0, fourth column probability for class 1

# project_root='/home/yz/autofocus/'
# os.chdir(project_root)

import numpy as np
import os
import caffe
import argparse
import math
from google.protobuf import text_format
import csv

# parse input
parser = argparse.ArgumentParser(
    description = 'generate prediction for given annotation file')
parser.add_argument('--MODEL_DEF', type = str,
                    default = "")
parser.add_argument('--MODEL_WEIGHTS', type = str,
                    default = "")
parser.add_argument('--OUTPUT', type = str,
                    default = "prediction.txt")
parser.add_argument('--GPU_ID', type = int, default = 0)
parser.add_argument('--MAX_NUMFILES', type = int, default = 99999999)
parser.add_argument('--PHASE', type = str, default = 'TEST')

parsed = parser.parse_args()
model_def = parsed.MODEL_DEF
model_weights = parsed.MODEL_WEIGHTS
destination = parsed.OUTPUT
gpuid = parsed.GPU_ID
maxnumfiles  =parsed.MAX_NUMFILES
phase = parsed.PHASE.lower()

# set caffe
caffe.set_mode_gpu()
caffe.set_device(gpuid)
net = None
if phase == 'test':
    net = caffe.Net(model_def, model_weights, caffe.TEST)
elif phase == 'train':
    net = caffe.Net(model_def, model_weights, caffe.TRAIN)
else:
    print('wrong phase')

# search for mean and prob layer
presoftmax = None
label = None
for v in net.bottom_names.keys():
    if v == "softmaxloss":
        presoftmax = net.bottom_names[v][0]
        label = net.bottom_names[v][1]

# get net parameters
netparam = caffe.proto.caffe_pb2.NetParameter()
with open(model_def, 'r') as model:
    text_format.Merge(str(model.read()), netparam)

# find eps value
for layer in netparam.layer:
    if layer.name == 'data':
        input_txt = layer.image_data_param.source

batch_size = net.blobs[label].data.shape[0]

# read file
filenames = []
labels = []
with open(input_txt, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ' ')
    for row in csvreader:
        filenames.append(row[0])
        labels.append(row[1])

# inference to get data and probability
prob0 = []
prob1 = []
numfiles = min(len(filenames), maxnumfiles)
num_batches = numfiles/batch_size+1
for i in range(num_batches):
    net.forward()
    tmp = net.blobs[presoftmax].data
    tmp0 = np.exp(tmp[:, 0]) / (np.exp(tmp[:, 0]) + np.exp(tmp[:, 1]))
    prob0 = np.append(prob0, tmp0)
prob0 = prob0[0:numfiles]
prob1 = 1-prob0

# write file
with open(destination, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ' ')
    for i in range(numfiles):
      csvwriter.writerow([filenames[i], labels[i], prob0[i], prob1[i]])