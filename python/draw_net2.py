# draw a box of the network

caffe_root = '/home/yz/caffe-yao/' 
import sys
sys.path.insert(0, caffe_root + 'python')

import argparse
import caffe
from google.protobuf import text_format
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource
import numpy as np

# parse input
parser = argparse.ArgumentParser(
    description = 'Draw the architecture of a new')
parser.add_argument('--MODEL_DEF', type = str,
                    default = "models/probnet14/test_1.prototxt")
parser.add_argument('--MODEL_WEIGHTS', type = str,
                    default = "models/probnet14/stage_1_final_1.caffemodel")
parser.add_argument('--OUTPUT', type = str,
                    default = "drawnet.png")
parser.add_argument('--PHASE', type = str, default = 'TEST')

parsed = parser.parse_args()
model_def = parsed.MODEL_DEF
model_weights = parsed.MODEL_WEIGHTS
destination = parsed.OUTPUT
phase = parsed.PHASE.lower()

net = None
if phase == 'test':
    net = caffe.Net(model_def, model_weights, caffe.TEST)
elif phase == 'train':
    net = caffe.Net(model_def, model_weights, caffe.TRAIN)
else:
    print('wrong phase')

# get net parameters
netparam = caffe.proto.caffe_pb2.NetParameter()
with open(model_def, 'r') as model:
    text_format.Merge(str(model.read()), netparam)
    
def unique_list(layers):
    used = []
    layers = [x for x in layers if x not in used and (used.append(x) or True)]
    return layers
    
def get_name(layers):
    if isinstance(layers, list):
        names=[]
        for layer in layers:
            names.append(str(layer.name))
        return names
    else:
        return str(layers.name)

def get_type(layers):
    if isinstance(layers, list):
        types=[]
        for layer in layers:
            types.append(str(layers.type))
        return types
    else:
        return str(layers.type)

def get_tops(layer):
    topnames=[]
    for top in layer.top:
        topnames.append(str(top))
    return topnames
    
def get_bottoms(layer):
    bottomnames=[]
    for bottom in layer.bottom:
        bottomnames.append(str(bottom))
    return bottomnames
    
def get_layer(netparam, layername):
    for layer in netparam.layer:
        if layername == str(layer.name):
            return layer
    return None
    
def get_layers_with_bottom(netparam, bottomname):
    layers=[]
    for layer in netparam.layer:
        if bottomname in get_bottoms(layer):
                # will skip in-place layer and layer with loss_weight zero
            if layer.bottom[0] != layer.top[0] and \
               (not layer.loss_weight or np.sum(layer.loss_weight)>0.0):
                layers.append(layer)
    return layers
    
def get_next_layers(netparam, layer, ignore=[]):
    layers=[]
    for topname in get_tops(layer):
        if topname not in ignore:
            layers.extend(get_layers_with_bottom(netparam, topname))
    return unique_list(layers)
    
def get_sorted_net(netparam):
    layers=[netparam.layer[0]]
    layerid=0
    sortedlayers=[]
    while layers:
        sortedlayers.append(layers)
        nextlayers=[]
        for layer in layers:
            nextlayers.extend(get_next_layers(netparam, layer, ignore=['label']));
        layers=unique_list(nextlayers)
        layerid +=1
    return sortedlayers

def check_connections(netparam, bottomlayer, toplayer):
    for top in get_tops(bottomlayer):
        for bot in get_bottoms(toplayer):
            if top == bot:
                return True
    return False

def get_connections(netparam, sortedlayers):
    connections=[]
    for ilayer in range(len(sortedlayers)-1):
        bottomlayers=sortedlayers[ilayer]
        toplayers=sortedlayers[ilayer+1]
        nb=len(bottomlayers)
        nt=len(toplayers)
        tmp=np.zeros((nb,nt))
        for ib, bottomlayer in zip(range(nb), bottomlayers):
            for it, toplayer in zip(range(nt), toplayers):
                tmp[ib,it]=check_connections(netparam, bottomlayer, toplayer)
        connections.append(tmp)
    return connections

def get_label(layer):
    layertype=str(layer.type)
    if layertype == 'Convolution':
        label='Conv ,'
        param=layer.convolution_param
        kernel=param.kernel_size[0]
        label+=str(param.num_output)+', '+str(kernel)+'X'+str(kernel)
    elif layertype in ['CyclicSlice','CyclicPool','CyclicRoll']:
        label=layertype
    elif 'Data' in layertype:
        label='Data'
    elif layertype == 'Pooling':
        method = layer.pooling_param.pool
        stride = layer.pooling_param.stride
        if method == 0:
            label='MaxPool ,'+str(stride)+'X'+str(stride)
        if method == 1:
            label='Global MeanPool'
    elif 'Loss' in layertype:
        label=layertype
    elif layertype == 'InnerProduct':
        label='Dense, '+str(layer.inner_product_param.num_output)
    else:
        label=''
    return label
    
def get_color(layer):
    layertype=str(layer.type)
    if layertype == 'Convolution':
        color='b'
    elif layertype in ['CyclicSlice','CyclicPool','CyclicRoll']:
        color='c'
    elif 'Data' in layertype:
        color='g'
    elif layertype == 'Pooling':
        color='m'
    elif 'Loss' in layertype:
        color='y'
    elif layertype == 'InnerProduct':
        color='r'
    else:
        color='k'
    return color
    
sortedlayers = get_sorted_net(netparam)
#connections = get_connections(netparam, sortedlayers)

bw=3
bh=0.4
nh = len(sortedlayers)
nw=0
for x in sortedlayers:
    nw=max(nw,len(x))
pad=0.005
border=0.01

plt.figure(figsize=(nw*bw,nh*bh), dpi=600)
ax=plt.axes()

for nhi, layers in zip(range(nh), sortedlayers):
    thisnw = len(layers)
    for nwi, layer in zip(range(thisnw),layers):
        x=(nw-thisnw)/2./nw+nwi/float(nw)+border
        y= 1-(nhi+1)/float(nh)+border
        wx=1.0/nw-border*2
        wy=1.0/nh-border*2
        fancybox = mpatches.FancyBboxPatch(
            [x,y], wx, wy,
            boxstyle=mpatches.BoxStyle("Round", pad=pad),
            color=get_color(layer))
        label=get_label(layer)
        ax.text(x+wx/2., y+wy/2.,label,
            size=12, transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='center',
            )

        ax.add_patch(fancybox)

ls = LightSource(azdeg=315, altdeg=45)
plt.axis('off')
plt.savefig(destination, dpi=600)
