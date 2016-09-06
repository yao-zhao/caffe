# generate predicted probability for binary classes
# output probability for class 0 and 1
# output: first column label,
# second column probability for class 0
# thirth column probability for class 1
# the label file will alow white space, so that last column is label

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
parser.add_argument('--INPUT', type = str,
                    default = None)
parser.add_argument('--OUTPUT', type = str,
                    default = "prediction.txt")
parser.add_argument('--GPU_ID', type = int, default = 0)
parser.add_argument('--MAX_NUMFILES', type = int, default = 99999999)
parser.add_argument('--PHASE', type = str, default = 'TEST')

parsed = parser.parse_args()
model_def = parsed.MODEL_DEF
model_weights = parsed.MODEL_WEIGHTS
inputfile = parsed.INPUT
destination = parsed.OUTPUT
gpuid = parsed.GPU_ID
maxnumfiles  =parsed.MAX_NUMFILES
phase = parsed.PHASE.lower()

if phase == 'test':
    phase = caffe.TEST
elif phase == 'train':
    phase = caffe.TRAIN
else:
    print('wrong phase')

# set caffe
caffe.set_mode_gpu()
caffe.set_device(gpuid)
net = None
net = caffe.Net(model_def, model_weights, phase)

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

# find source file value
for layer in netparam.layer:
    if layer.name == 'data':
        if inputfile is not None:
            layer.image_data_param.source = inputfile
        else:
            inputfile = layer.image_data_param.source
        layer.image_data_param.shuffle = False

with open('tempnet.prototxt', 'w+') as f:
    f.write(str(netparam))

net = None
net = caffe.Net('tempnet.prototxt', model_weights, phase)

batch_size = net.blobs[label].data.shape[0]

# read file
filenames = []
labels = []
with open(inputfile, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ' ')
    for row in csvreader:
        filenames.append(' '.join(row[0:-1]))
        labels.append(row[-1])

# inference to get data and probability
prob0 = []
prob1 = []
labels2 = []
numfiles = min(len(filenames), maxnumfiles)
num_batches = numfiles/batch_size+1
for i in range(num_batches):
    net.forward()
    tmp = net.blobs[presoftmax].data
    tmp0 = np.exp(tmp[:, 0]) / (np.exp(tmp[:, 0]) + np.exp(tmp[:, 1]))
    prob0 = np.append(prob0, tmp0)
    labels2 = np.append(labels2, net.blobs['label'].data)
prob0 = prob0[0:numfiles]
prob1 = 1-prob0

os.remove('tempnet.prototxt')

# write file
with open(destination, 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter = ' ')
    for i in range(numfiles):
      csvwriter.writerow([filenames[i], labels[i], labels2[i],\
        prob0[i], prob1[i]])
