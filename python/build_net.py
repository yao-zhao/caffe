# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 22:07:56 2016
collection of quick helper to build layers
@author: yz
"""
import os
import caffe
from caffe import layers as L, params as P

class BuildNet:
        
    # ini def
    def __init__(self, func, name = ''):
        self.index = 1
        self.bottom = None
        self.label = None
        self.net = caffe.NetSpec()
        self.phase = 'train'
        self.func = func
        
    # increase index
    def increase_index(self):
        self.index += 1
        
    # change the bottom
    def set_bottom(self, name):
        self.bottom = getattr(self.net, name)
    
    # reset net
    def reset(self):
        self.index = 1
        self.net = None
        self.net = caffe.NetSpec()

    # set data layer
    def add_lmdb(self, transformer_dict = None, batch_size = 32,
                 backend = P.Data.LMDB, source_path = 'data/'):
        if self.phase == 'train':
            self.bottom, self.label = L.Data(batch_size = batch_size, 
                    backend = backend, source = source_path + 'train_lmdb',
                    transform_param = transformer_dict,
                    ntop=2, include=dict(phase=caffe.TRAIN))
            self.net.data = self.bottom
            self.net.label = self.label      
        elif self.phase == 'test':
            self.bottom, self.label = L.Data(batch_size = batch_size, 
                    backend = backend, source = source_path + 'val_lmdb',
                    transform_param = transformer_dict,
                    ntop=2, include=dict(phase=caffe.TEST))
            self.net.data = self.bottom
            self.net.label = self.label      
        elif self.phase == 'deploy':
            self.net.data = self.bottom

    # set image data layer
    def add_image(self, transformer_dict = None, batch_size = 32,
                 source_path = 'data/', root_folder = 'data/',
                 shuffle = True, is_color = True, height = None, width = None):
        # probe image data dimension
        if not height or not width:
            tmpnet = caffe.NetSpec()
            tmpnet.data, tmpnet.label = L.ImageData(source = source_path + 'train.txt',
                root_folder = root_folder, is_color = is_color,
                batch_size= 1, ntop = 2)
            with open('tmpnet.prototxt', 'w+') as f:
                f.write(str(tmpnet.to_proto()))
            nb, nc, h, w = caffe.Net('tmpnet.prototxt', caffe.TRAIN).\
                blobs['data'].data.shape
            if not height:
                height = h
            if not width:
                width = w
            os.remove('tmpnet.prototxt')
        # add layer
        if self.phase == 'train':
            self.bottom, self.label = L.ImageData(batch_size = batch_size, 
                    source = source_path + 'train.txt',
                    root_folder = root_folder, is_color = is_color,
                    transform_param = transformer_dict, ntop=2,
                    new_height = height, new_width = width,
                    include=dict(phase=caffe.TRAIN))
            self.net.data = self.bottom
            self.net.label = self.label    
        elif self.phase == 'test':
            self.bottom, self.label = L.ImageData(batch_size = batch_size, 
                    source = source_path + 'val.txt',
                    root_folder = root_folder, is_color = is_color,
                    shuffle = shuffle,
                    transform_param = transformer_dict, ntop=2,
                    new_height = height, new_width = width,
                    include=dict(phase=caffe.TEST))
            self.net.data = self.bottom
            self.net.label = self.label   
        elif self.phase == 'deploy':
            self.net.data = self.bottom
            

    # add pooling layer of 2
    def add_maxpool_2(self):
        self.bottom = L.Pooling(self.bottom,
            kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
        setattr(self.net, 'pool'+str(self.index), self.bottom)
        self.index += 1
    
    # add final mean pool
    def add_meanpool_final(self):
        self.bottom = L.Pooling(self.bottom, global_pooling = True,
            stride = 1, pool = P.Pooling.AVE)
        setattr(self.net, 'pool'+str(self.index), self.bottom)
        self.index += 1
    
    # add fc
    def add_fc(self, num_output, lr = 1):
        if self.phase == 'train' or self.phase == 'test':
            self.bottom = L.InnerProduct(self.bottom,
                num_output = num_output, param=[dict(lr_mult= lr)],
                weight_filler=dict(type='xavier'))
        elif self.phase == 'deploy':
            self.bottom = L.InnerProduct(self.bottom,
                num_output = num_output)
        setattr(self.net, 'fc'+str(self.index), self.bottom)
        self.index += 1
    
    # add softmax
    def add_softmax(self, loss_weight = 1):
        if self.phase == 'train' or self.phase == 'test':
            softmax = L.SoftmaxWithLoss(self.bottom, self.label,
                                        loss_weight = loss_weight)
            accuracy = L.Accuracy(self.bottom, self.label)
            setattr(self.net, 'loss', softmax)
            setattr(self.net, 'accuracy', accuracy)
        elif self.phase =='deploy':
            softmax = L.Softmax(self.bottom)
            setattr(self.net, 'prob', softmax)

    # add softmax
    def add_euclidean(self, loss_weight = 1):
        if self.phase == 'train' or self.phase == 'test':
            euclidean = L.EuclideanLoss(self.bottom, self.label,
                                      loss_weight = loss_weight)
            setattr(self.net, 'loss', euclidean)
            
    # convolutional layer 
    def add_conv(self, num_output, lr=1, kernel_size=3, pad=1, stride=1):
        if self.phase == 'train' or self.phase == 'test':
            self.bottom = L.Convolution(self.bottom, 
                kernel_size=kernel_size, pad=pad, 
                stride=stride, num_output=num_output,
                weight_filler=dict(type='xavier'),
                param=dict(lr_mult= lr), bias_term=False)
        elif self.phase == 'deploy':
            self.bottom =  L.Convolution(self.bottom, 
                kernel_size=kernel_size, pad=pad, 
                stride=stride, num_output=num_output,
                bias_term=False)
        else:
            print "phase not supported"
        setattr(self.net, 'conv'+str(self.index), self.bottom)
        
    # add a typical block       
    def add_normal_block(self, num_output, lr = 1):
        self.add_conv(num_output, lr = lr)
        self.add_batchnorm()
        self.add_scale(lr = lr)
        self.add_relu()
        self.index += 1
    
    # add batch normalization
    def add_batchnorm(self):
        if self.phase == 'train':
            self.bottom = L.BatchNorm(self.bottom,
                param=[dict(lr_mult= 0),dict(lr_mult= 0),dict(lr_mult= 0)],
                batch_norm_param=dict(use_global_stats=False),
                in_place=True)
        elif self.phase == 'test':
            self.bottom = L.BatchNorm(self.bottom,
                param=[dict(lr_mult= 0),dict(lr_mult= 0),dict(lr_mult= 0)],
                batch_norm_param=dict(use_global_stats=True),
                in_place=True)
        elif self.phase == 'deploy':
            self.bottom = L.BatchNorm(self.bottom,
                batch_norm_param=dict(use_global_stats=True),
                in_place=True)
        setattr(self.net, 'bn'+str(self.index), self.bottom)
    
    # add scale function
    def add_scale(self, lr = 1):
        if self.phase == 'train' or self.phase == 'test':
            self.bottom = L.Scale(self.bottom,
                  param=[dict(lr_mult= lr), dict(lr_mult= lr)],
                  bias_term=True,in_place=True,
                  bias_filler=dict(type='constant',value=0))
        else:
            self.bottom = L.Scale(self.bottom,
                  param=[dict(lr_mult= lr), dict(lr_mult= lr)],
                  bias_term=True,in_place=True)
        setattr(self.net, 'scale'+str(self.index), self.bottom)
        
    # ReLU 
    def add_relu(self):
        self.bottom = L.ReLU(self.bottom, in_place = True)
        setattr(self.net, 'relu'+str(self.index), self.bottom)
    
    # Parameterized ReLU 
    def add_prelu(self, lr = 1):
        self.bottom = L.PReLU(self.bottom, in_place = True, 
              param=[dict(lr_mult= lr)])
        setattr(self.net, 'prelu'+str(self.index), self.bottom)
        
    # save
    def save(self, name = 'net', savepath = 'models/'):
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        for phase in ['train','test','deploy']:
            self.reset()            
            self.phase = phase
            self.func(self)
            with open(savepath+name+'/'+self.phase+'.prototxt', 'w+') as f:
                f.write('name: \"'+name+'\"\n')
                if phase == 'deploy':
                    numbatch, numchannel, height, width = caffe.Net(
                        savepath+name+'/train.prototxt',
                        caffe.TRAIN).blobs['data'].shape
                    f.write('input: \"data\"\n')
                    f.write('input_dim: '+str(numbatch)+'\n')
                    f.write('input_dim: '+str(numchannel)+'\n')
                    f.write('input_dim: '+str(height)+'\n')
                    f.write('input_dim: '+str(width)+'\n')
                f.write(str(self.net.to_proto() ))
                print self.phase+' net writing finished!' 