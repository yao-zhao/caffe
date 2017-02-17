# draw a box of the network
caffe_root = '/home/yz/caffe3/' 
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
#                    default = "/home/yz/autofocus/models/probnet12/test_1.prototxt")
                    default = "models/segnet5f/test_0.prototxt")
parser.add_argument('--OUTPUT', type = str,
                    default = "drawnet.pdf")
parser.add_argument('--PHASE', type = str, default = 'TEST')

parsed = parser.parse_args()
model_def = parsed.MODEL_DEF
destination = parsed.OUTPUT
phase = parsed.PHASE.lower()

# get net parameters
netparam = caffe.proto.caffe_pb2.NetParameter()
with open(model_def, 'r') as model:
    text_format.Merge(str(model.read()), netparam)
    
# get a unique list of layers
def unique_list(layers):
    used = []
    layers = [x for x in layers if x not in used and (used.append(x) or True)]
    return layers

# get names of layer
def get_name(layers):
    if isinstance(layers, list):
        names=[]
        for layer in layers:
            names.append(str(layer.name))
        return names
    else:
        return str(layers.name)

# get types of layers
def get_type(layers):
    if isinstance(layers, list):
        types=[]
        for layer in layers:
            types.append(str(layers.type))
        return types
    else:
        return str(layers.type)

# get tops of 1 layer
def get_tops(layer):
    topnames=[]
    for top in layer.top:
        topnames.append(str(top))
    return topnames

# get bottoms of 1 layer
def get_bottoms(layer):
    bottomnames=[]
    for bottom in layer.bottom:
        bottomnames.append(str(bottom))
    return bottomnames

# get the pointer to the layer with layername
def get_layer(netparam, layername):
    for layer in netparam.layer:
        if layername == str(layer.name):
            return layer
    return None

# get all the layers that has bottom with bottomname
# when search for bottoms,
# ignore layers with 0 loss weight
# ignroe in-place layer
# ignore second bottoms of upsample, loss layers and so on
def get_layers_with_bottom(netparam, bottomname):
    layers=[]
    for layer in netparam.layer:
        layerbottoms = get_bottoms(layer)
        for layerbottom, index in zip(layerbottoms, range(len(layerbottoms))):
            if bottomname == layerbottom \
                and not (str(layer.type) == 'Upsample' and index == 1) \
                and not ('Loss' in str(layer.type) and index in [1, 2]) \
                and layer.bottom[0] != layer.top[0] \
                and (not layer.loss_weight or np.sum(layer.loss_weight)>0.0):
                layers.append(layer)
    return layers

# get all next layers with the option to ignore certain names
# ignores are case sensitive
def get_next_layers(netparam, layer, ignores=[]):
    layers=[]
    for topname in get_tops(layer):
        for newlayer in get_layers_with_bottom(netparam, topname):            
            doignore = False            
            for ignore in ignores:
                if ignore in str(newlayer.name):
                    doignore = True 
            if not doignore:
                layers.append(newlayer)
    return unique_list(layers)

# get the net sorted, each level is a list of layers
def get_sorted_net(netparam):
    layers=[netparam.layer[0]]
    layerid=0
    sortedlayers=[]
    while layers:
        sortedlayers.append(layers)
        nextlayers=[]
        for layer in layers:
            nextlayers.extend(get_next_layers(netparam, layer,
                                              ignores=['label', 'accuracy']));
        layers=unique_list(nextlayers)
        layerid +=1
    return sortedlayers

# check if connection exist between bottom and top layers
def check_connections(netparam, bottomlayer, toplayer):
    for top in get_tops(bottomlayer):
        for bot in get_bottoms(toplayer):
            if top == bot:
                return True
    return False

# get all connections in the sorted net
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
    
# get all arrows
def get_arrows(netparam, sortedlayers):
    arrows = []
    target_layers = [layer for layer in netparam.layer \
        if str(layer.type) == 'Upsample']
    for target_layer in target_layers:
        arrow = np.zeros((4)).astype(np.int)
        bottomname = str(target_layer.bottom[1])
        topname = str(target_layer.name)
        for layers, irow in zip(sortedlayers, range(len(sortedlayers))):
            for layer, icol in zip(layers, range(len(layers))):
                if len(layer.top)==2 and str(layer.top[1]) == bottomname:
                    arrow[0] = irow
                    arrow[1] = icol
                elif str(layer.name) == topname:
                    arrow[2] = irow
                    arrow[3] = icol
        arrows.append(arrow)
    return arrows

# decide how to label layers on the graph
def get_label(layer):
    layertype=str(layer.type)
    if layertype == 'Convolution':
        label='Conv, '
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
            label='MaxPool, '+str(stride)+'X'+str(stride)
        if method == 1:
            label='Global MeanPool'
    elif 'Loss' in layertype:
        label=layertype
    elif layertype == 'InnerProduct':
        label='Dense, '+str(layer.inner_product_param.num_output)
    elif layertype == 'Upsample':
        scale = layer.upsample_param.scale
        label = 'Upsample, '+str(scale)+'X'+str(scale)
    else:
        label=''
        print('can not find label for layer: type - '+layertype+' - name -'
            + str(layer.name))
    return label

# decide color for each layer
def get_color(layer):
    layertype=str(layer.type)
    if layertype == 'Convolution':
        color='b'
    elif layertype in ['CyclicSlice','CyclicPool','CyclicRoll']:
        color='c'
    elif 'Data' in layertype:
        color='g'
    elif layertype == 'Pooling' or layertype == 'Upsample':
        color='m'
    elif 'Loss' in layertype:
        color='y'
    elif layertype == 'InnerProduct':
        color='r'
    else:
        color='k'
    return color
    
sortedlayers = get_sorted_net(netparam)
connections = get_connections(netparam, sortedlayers)
arrows = get_arrows(netparam, sortedlayers)

bw=5
bh=0.4
nh = len(sortedlayers)
nw=0
for x in sortedlayers:
    nw=max(nw,len(x))
pad=0.005
border=0.01

plt.figure(figsize=(nw*bw,nh*bh), dpi=600)
ax=plt.axes()

wscale = 1
wscale_step = 0.10

xs = np.zeros((nh, nw))
ys = np.zeros((nh, nw))

for nhi, layers in zip(range(nh), sortedlayers):
    thisnw = len(layers)
    for nwi, layer in zip(range(thisnw),layers):
        # change width
        if str(layer.type) == 'Upsample':
            wscale += wscale_step        
        # calculate coordinates
        y= 1-(nhi+1)/float(nh)+border
        wx=1.0/nw*wscale-border*2
        wy=1.0/nh-border*2
        #x=(nw-thisnw)/2./nw+nwi/float(nw)+border
        x= 1/2 - 1/nw*thisnw*wscale/2 + (nwi)/nw*wscale+border
        xs[nhi, nwi] = x+wx
        ys[nhi, nwi] = y+wy
        # draw box and text
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
        if str(layer.type) == 'Pooling':
            wscale -= wscale_step
        ax.add_patch(fancybox)

# plot arrows
for arrow in arrows:
    ax.arrow(xs[arrow[0], arrow[1]]-border, ys[arrow[0], arrow[1]]-1/nh+border, 
             0, 
             ys[arrow[2], arrow[3]] - ys[arrow[0], arrow[1]]+2/nh-2*border, 
             head_width=2*border, head_length=border, fc='k', ec='k')


ls = LightSource(azdeg=315, altdeg=45)
plt.axis('off')
plt.savefig(destination, dpi=600)
