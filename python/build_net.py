# -*- coding: utf-8 -*-
"""
collection of quick helper to build layers
"""
from __future__ import print_function

import os
os.environ["GLOG_minloglevel"] = "2"
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import numpy as np

class BuildNet:
    # ini def
    def __init__(self, func, name = 'net', savepath = 'models/',\
                 caffe_path = '~/caffe-yao/'):
        self.func = func
        self.model_path = savepath + name +'/'
        self.name = name
        self.reset()
        self.number_stages = len(self.solvers)
        self.caffe_path = caffe_path
        print("initialization done")

# common task
################################################################################
    # increase index
    def increase_index(self):
        self.index += 1

    # change the bottom
    def set_bottom(self, name):
        if type(name) in ['str', 'unicode']:
            self.bottom = getattr(self.net, name)
        else:
            self.bottom = name

    # reset net
    def reset(self, phase = 'train', stage = None):
        self.index = 0
        self.net = None
        self.net = caffe.NetSpec()
        self.solvers = []
        self.stage = stage
        self.stage_iters = []
        self.bottom = None
        self.label = None
        self.phase = phase
        self.unpool_array = []
        self.shortcut_array = []
        self.adversarial_stage = 'none'
        self.input_dim = []
        self.func(self)

    # check stage
    def check_stage(self, stage):
        return (stage is None or self.stage is None or \
                np.any(np.equal(stage, self.stage))) and \
                (self.adversarial_stage is not 'discrimitive' \
                or self.phase is not 'deploy')


# function groups
################################################################################
    # add a typical block
    def add_normal_block(self, num_output, lr = 1, stage = None):
        self.add_conv(int(num_output), lr = lr, stage = stage)
        self.add_batchnorm(stage = stage)
        self.add_scale(lr = lr, stage = stage)
        self.add_relu(stage = stage)

    # add a typical block
    def add_deconv_block(self, num_output, lr = 1, stage = None):
        self.add_deconv(num_output, lr = lr, stage = stage)
        self.add_batchnorm(stage = stage)
        self.add_scale(lr = lr, stage = stage)
        self.add_relu(stage = stage)

    # add a prelu block
    def add_prelu_block(self, num_output, lr = 1, stage = None):
        self.add_conv(num_output, lr = lr, stage = stage)
        self.add_batchnorm(stage = stage)
        self.add_scale(lr = lr, stage = stage)
        self.add_prelu(stage = stage)

    # add a conv pool block
    def add_convpool_block(self, num_output, lr = 1, stage = None):
        self.add_conv(num_output, lr = lr, stride = 2, stage = stage)
        self.add_batchnorm(stage = stage)
        self.add_scale(lr = lr, stage = stage)
        self.add_relu(stage = stage)

    # add a prelu conv pool block
    def add_prelu_convpool_block(self, num_output, lr = 1, stage = None):
        self.add_conv(num_output, lr = lr, stride = 2, stage = stage)
        self.add_batchnorm(stage = stage)
        self.add_scale(lr = lr, stage = stage)
        self.add_prelu(stage = stage)

    # add bottle neck block
    def add_bottleneck_block(self, num_output, num_bottleneck=None,
                             first_stride=1, lr = 1, stage = None):
        if num_bottleneck is None:
            num_bottleneck = num_output/4
        self.add_conv(num_bottleneck, lr=lr, stage=stage, kernel_size=1,
                      pad=0, stride=first_stride)
        self.add_conv(num_bottleneck, lr=lr, stage=stage)
        self.add_conv(num_output, lr=lr, stage=stage, kernel_size=1,
                      pad=0, stride=1)

    # add resnet block

# input layers
################################################################################
    def choose_input_param(self, batch_size, test_batch_size,\
            transformer_dict, test_transformer_dict ,\
            source_path, source_train, source_test, shuffle, test_shuffle):
        if self.phase == 'train':
            source = source_path + source_train
        if self.phase == 'test':
            if test_batch_size is not None:
                batch_size = test_batch_size
            if test_transformer_dict is not None:
                transformer_dict = test_transformer_dict
            shuffle = test_shuffle
            source = source_path + source_test
        if self.phase == 'deploy':
            batch_size = 1
            source = source_path + source_train
        return batch_size, transformer_dict, source, shuffle

    def probe_input_dim(self, inputfuc, source, root_folder,\
            is_color, transformer_dict):
        # probe image data dimension
        tmpnet = caffe.NetSpec()
        tmpnet.data, tmpnet.label = inputfuc(source = source,
            transform_param = transformer_dict,
            root_folder = root_folder, is_color = is_color,
            batch_size = 1, ntop = 2)
        with open('tmpnet.prototxt', 'w+') as f:
            f.write(str(tmpnet.to_proto()))
        nb, nc, h, w = caffe.Net('tmpnet.prototxt', caffe.TRAIN).\
            blobs['data'].data.shape
        os.remove('tmpnet.prototxt')
        return nb, nc, h, w

    # set data layer
    def add_input(self, transformer_dict = None, batch_size = 32,
                 test_batch_size = None, data_dim = [1, 1, 1],
                 label_dims = [[1]]):
        if self.phase == 'test':
            if test_batch_size is not None:
                batch_size = test_batch_size
        if self.phase == 'deploy':
            batch_size = 1
        self.bottom = L.Input(input_param =
                dict(shape = dict(dim = [batch_size]+data_dim)))
        self.net.data = self.bottom
        numlabels = len(label_dims)
        for ilabel in range(numlabels):
            if not self.phase == 'deploy':
                label = L.Input(input_param =
                    dict(shape = dict(dim = [batch_size]+label_dims[ilabel])))
                setattr(self.net, 'label'+str(ilabel), label)
        self.input_dim = [batch_size]+data_dim
        return self.bottom

    def add_lmdb(self, transformer_dict = None, batch_size = 32,
                 test_batch_size = None,
                 backend = P.Data.LMDB, source_path = 'data/'):
        # temporarily broken
        # if test_batch_size is None:
        #     test_batch_size = batch_size
        # if self.phase == 'train':
        #     self.bottom, self.label = L.Data(batch_size = batch_size, 
        #             backend = backend, source = source_path + 'train_lmdb',
        #             transform_param = transformer_dict,
        #             ntop = 2, include = dict(phase = caffe.TRAIN))
        # elif self.phase == 'test':
        #     self.bottom, self.label = L.Data(batch_size = test_batch_size,
        #             backend = backend, source = source_path + 'val_lmdb',
        #             transform_param = transformer_dict,
        #             ntop = 2, include = dict(phase = caffe.TEST))
        # elif self.phase == 'deploy':
        #     self.bottom = L.Input(input_param = dict(shape = dict(dim =
        #                          [1, nc, height, width])))
        # self.net.data = self.bottom
        # if not self.phase == 'deploy':
        #     self.net.label = self.label
        # self.input_dim = [dim0]+data_dim
        return self.bottom

    # set image data layer
    def add_image(self, transformer_dict = None, batch_size = 32,
                 test_batch_size = None,
                 source_path = 'data/', root_folder = 'data/',
                 source_train = 'train.txt', source_test = 'val.txt',
                 label_scale = 1, test_transformer_dict = None,
                 shuffle = True, test_shuffle = False,
                 is_color = True, height = None, width = None):
        batch_size, transformer_dict, source, shuffle =\
            self.choose_input_param(batch_size, test_batch_size,\
            transformer_dict, test_transformer_dict ,\
            source_path, source_train, source_test, shuffle, test_shuffle)
        nb, nc, h, w = self.probe_input_dim(L.ImageData, source, root_folder,\
            is_color, transformer_dict)
        if self.phase == 'train' or self.phase == 'test':
            if height and width:
                self.bottom, self.label = L.ImageData(batch_size = batch_size,
                        source = source,
                        root_folder = root_folder, is_color = is_color,
                        shuffle = shuffle, label_scale = label_scale,
                        transform_param = transformer_dict, ntop = 2,
                        new_height = height, new_width = width)
            else:
                self.bottom, self.label = L.ImageData(batch_size = batch_size,
                        source = source,
                        root_folder = root_folder, is_color = is_color,
                        shuffle = shuffle, label_scale = label_scale,
                        transform_param = transformer_dict, ntop = 2)
        elif self.phase == 'deploy':
            self.bottom = L.Input(input_param = dict(shape = dict(dim =
                                 [batch_size, nc, h, w])))
        self.net.data = self.bottom
        if not self.phase == 'deploy':
            self.net.label = self.label
        self.input_dim = [batch_size, nc, h, w]
        return self.bottom

    # set image data layer
    def add_image_box(self, batch_size=32, max_num_box=8,
            source_path='data/', root_folder='data/', test_batch_size=None,
            source_train='train.txt', source_test ='val.txt',
            shuffle = True, test_shuffle=False,
            is_color = True, height = None, width = None):
        batch_size, transformer_dict, source, shuffle =\
            self.choose_input_param(batch_size, test_batch_size,\
            transformer_dict, test_transformer_dict ,\
            source_path, source_train, source_test, shuffle, test_shuffle)
        nb, nc, h, w = self.probe_input_dim(L.add_image_box, source,\
            root_folder, is_color, transformer_dict)
        if self.phase == 'train' or self.phase == 'test':
            if height and width:
                self.bottom, self.label = L.ImageBoxData(
                    batch_size=batch_size, max_num_box=max_num_box,
                    source=source, root_folder=root_folder,
                    is_color=is_color, shuffle=shuffle, ntop=2,
                    new_height = height, new_width = width)
            else:
                self.bottom, self.label = L.ImageBoxData(
                    batch_size=batch_size, max_num_box=max_num_box,
                    source=source, root_folder=root_folder,
                    is_color=is_color, shuffle=shuffle, ntop=2)
        elif self.phase == 'deploy':
            self.bottom = L.Input(input_param=dict(
                shape=dict(dim=[1, nc, h, w])))
        self.net.data = self.bottom
        if not self.phase == 'deploy':
            self.net.label = self.label
        self.input_dim = [batch_size, nc, h, w]
        return self.bottom

    # set image data layer
    def add_dense_image(self, transformer_dict = dict(), batch_size = 32,
                test_batch_size = None, 
                source_path = 'data/', root_folder = 'data/',
                source_train = 'train.txt', source_test = 'val.txt',
                test_transformer_dict = None,
                shuffle = True, is_color = True, test_shuffle = False):
        batch_size, transformer_dict, source, shuffle =\
            self.choose_input_param(batch_size, test_batch_size,\
            transformer_dict, test_transformer_dict ,\
            source_path, source_train, source_test, shuffle, test_shuffle)
        nb, nc, h, w = self.probe_input_dim(L.DenseImageData,\
            source, root_folder, is_color, transformer_dict)
        if self.phase == 'train' or self.phase == 'test':
                self.bottom, self.label = L.DenseImageData(
                        batch_size = batch_size, source = source,
                        root_folder = root_folder, is_color = is_color,
                        shuffle = shuffle,
                        transform_param = transformer_dict, ntop = 2)
        elif self.phase == 'deploy':
            self.bottom = L.Input(input_param = dict(shape = dict(dim =
                                 [batch_size, nc, h, w])))
        self.net.data = self.bottom
        if not self.phase == 'deploy':
            self.net.label = self.label
        self.input_dim = [batch_size, nc, h, w]
        return self.bottom

# pooling layers and upsample layers
################################################################################
    # add pooling layer of 2
    def add_maxpool_2(self, stage = None, use_unpool = False):
        if self.check_stage(stage):
            self.increase_index()
            if use_unpool is False:
                self.bottom = L.Pooling(self.bottom,
                    kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
                setattr(self.net, 'pool'+str(self.index), self.bottom)
            else:
                self.bottom, unpool = L.Pooling(self.bottom,
                    kernel_size = 2, stride = 2, pool = P.Pooling.MAX, ntop = 2)
                setattr(self.net, 'pool'+str(self.index), self.bottom)
                self.unpool_array.append(unpool)
        return self.bottom

    # add final mean pool
    def add_meanpool_final(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.Pooling(self.bottom, global_pooling = True,
                stride = 1, pool = P.Pooling.AVE)
            setattr(self.net, 'pool'+str(self.index), self.bottom)
        return self.bottom

    # add final mean pool
    def add_maxpool_final(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.Pooling(self.bottom, global_pooling = True,
                stride = 1, pool = P.Pooling.MAX)
            setattr(self.net, 'pool'+str(self.index), self.bottom)
        return self.bottom

    # add upsample layers
    def add_upsample(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.Upsample(self.bottom, self.unpool_array.pop(),
                scale = 2)
        return self.bottom

    # # add discretize label
    # def add_discrete_label(self, separator, stage = None):
    #     if self.check_stage(stage):
    #         self.label = L.Discretize(self.label,
    #             discretize_param=dict(separator=separator));

# output layers
################################################################################
    # add yolo layer
    def add_yolo(self, S_h, S_w, B, C=1,
            lr=1, weight_filler=dict(type='gaussian', std=0.1),
            lambda_coord=5, lambda_noobj=0.5,
            loss_weight=1, name='yolo', stage=None, label='label'):
        if self.check_stage(stage):
            lr = self.check_lr_mult(lr)
            base = self.bottom
            boxpre = L.InnerProduct(base, num_output=S_h*S_w*B*5,
                    param=[dict(lr_mult=lr), dict(lr_mult=lr)],
                    weight_filler=weight_filler,
                    bias_filler=dict(type='constant', value=0))
            boxsigmoid = L.Sigmoid(boxpre, in_place=True)
            box = L.Reshape(boxsigmoid, shape=dict(dim=[S_h, S_w, B, 5]), axis=1)
            setattr(self.net, 'yolo_box_pre', boxpre)
            setattr(self.net, 'yolo_box_sigmoid', boxsigmoid)
            setattr(self.net, 'yolo_box', box)
            if C > 1 :
                cprobpre = L.InnerProduct(base, numoutput=S_h*S_w*C,
                    param=[dict(lr_mult=lr), dict(lr_mult=lr)],
                    weight_filler=dict(type='xavier'),
                    bias_filler=dict(type='constant', value=0))
                cprob = L.Reshape(Cpre, shape=dict(dim=[S_h, S_w, C]), axis=1)
                cprobsoftmax = L.Softmax(C, axis=3, in_place=True)
                setattr(self.net, 'yolo_class_pre', cprobpre)
                setattr(self.net, 'yolo_class', cprob)
                setattr(self.net, 'yolo_class_softmax', cprobsoftmax)
            if self.phase == 'train' or self.phase == 'test':
                label = getattr(self.net, label)
                if C == 1 :
                    loss = L.YoloLoss(box, label,
                        lambda_coord=lambda_coord, lambda_noobj=lambda_noobj)
                elif C > 1:
                    loss = L.YoloLoss(box, label, cprob,
                        lambda_coord=lambda_coord, lambda_noobj=lambda_noobj)
                setattr(self.net, 'yolo_loss', loss)


    # add softmax layer
    def add_softmax(self, loss_weight = 1, name = 'softmax', stage = None,
                    class_weights = None, label = None, has_output = False):
        if self.check_stage(stage):
            if class_weights is None:
                loss_param = dict(weight_by_label_freqs = False)
            else:
                loss_param = dict(weight_by_label_freqs = True,
                    class_weighting = class_weights)
            if has_output:
                self.increase_index()
            if self.phase == 'train' or self.phase == 'test':
                if not label:
                    label = self.label
                else:
                    if isinstance(label, str):
                        label = getattr(self.net, label)
                if has_output:
                    loss, softmax = L.SoftmaxWithLoss(self.bottom, label,
                              loss_weight = [loss_weight, 0],
                              loss_param = loss_param, ntop = 2)
                else:
                    loss = L.SoftmaxWithLoss(self.bottom, label,
                                            loss_weight = loss_weight,
                                            loss_param = loss_param)
                accuracy = L.Accuracy(self.bottom, label)
                setattr(self.net, name+'loss'+str(self.index), loss)
                setattr(self.net, 'accuracy'+str(self.index), accuracy)
            elif self.phase == 'deploy':
                softmax = L.Softmax(self.bottom)
            if self.phase == 'deploy' or has_output:
                setattr(self.net, name+'prob'+str(self.index), softmax)
            if has_output:
                self.bottom = softmax
                return self.bottom


    # add softmax decay layer
    def add_softmax_decay(self, separator, loss_weight=1, name='softmax_decay_',
                          stage=None, decay_rate=1.0,):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                discrete_labels, bins = L.Discretize(self.label,
                    separator=separator, ntop=2);
                loss, softmax = L.SoftmaxWithDecayLoss(
                    self.bottom, discrete_labels,
                    loss_weight=[loss_weight, 0], rate=decay_rate, ntop=2)
                accuracy = L.Accuracy(softmax, self.label)
                predicted_labels = L.SoftmaxInterpolation(softmax, bins)
                euclidean = L.EuclideanLoss(predicted_labels, self.label,
                    loss_weight=0)
                setattr(self.net, 'discrete_label', discrete_labels)
                setattr(self.net, 'label_centers', bins)
                setattr(self.net, name+'loss', loss)
                setattr(self.net, name+'softmax', softmax)
                setattr(self.net, name+'accuracy', accuracy)
                setattr(self.net, name+'interpolated_label', predicted_labels)
                setattr(self.net, name+'euclidean_loss', euclidean)
            elif self.phase == 'deploy':
                softmax = L.Softmax(self.bottom)
                setattr(self.net, name+'softmax', softmax)

    # add softmax
    def add_euclidean(self, loss_weight = 1, name = 'euclidean',\
                    label = None, stage = None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                if not label:
                    label = self.label
                else:
                    label = getattr(self.net, label)
                euclidean = L.EuclideanLoss(self.bottom, label,
                                          loss_weight = loss_weight)
                setattr(self.net, name+'loss', euclidean)

    # add gaussian prob loss layer
    def add_gaussian_prob(self, mean, var, label = None,
        eps = 1e-2, loss_weight = 1, name = 'gaussianprob', stage = None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                if not label:
                    label = self.label
                else:
                    label = getattr(self.net, label)
                gaussianprob = L.GaussianProbLoss(mean, var, label,
                    gaussian_prob_loss_param = dict(eps = eps),
                    loss_weight = loss_weight)
                setattr(self.net, name+'loss', gaussianprob)

    # add gaussian prob loss layer
    def add_lorentzian_prob(self, mean, var, label = None,
        eps = 1e-2, loss_weight = 1, name = 'lorentzianprob', stage = None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                if not label:
                    label = self.label
                else:
                    label = getattr(self.net, label)
                prob = L.LorentzianProbLoss(mean, var, label,
                    gaussian_prob_loss_param = dict(eps = eps),
                    loss_weight = loss_weight)
                setattr(self.net, name+'loss', prob)

    # add roc loss layer to the output
    def add_roc(self, label = None, loss_weight = 1,\
        stage = None, name = 'roc', eps = 0.1):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                if not label:
                    label = self.label
                else:
                    label = getattr(self.net, label)
                roc = L.SoftmaxWithROCLoss(self.bottom, self.label,\
                    softmax_roc_loss_param = dict(eps = eps),\
                                            loss_weight = loss_weight)
                accuracy = L.Accuracy(self.bottom, self.label)
                setattr(self.net, name+'loss', roc)
                setattr(self.net, 'accuracy', accuracy)
            elif self.phase == 'deploy':
                softmax = L.Softmax(self.bottom)
                setattr(self.net, name+'prob', softmax)

    # add regression layer on label
    def add_regression_label(self, lb = 0, ub = 1, stage = None):
        if self.check_stage(stage) and \
                ( self.phase =='train' or self.phase == 'test'):
            label0, label1 = L.RegressionLabel(self.label,\
                name = 'regression_label',\
                regression_label_param = \
                dict(upper_bound = ub, lower_bound = lb), \
                ntop = 2)
            setattr(self.net, 'label0', label0)
            setattr(self.net, 'label1', label1)

# common building components
################################################################################
    # convolutional layer 
    def add_conv(self, num_output, lr=1, kernel_size=3, weight_filler=None,
        pad=1, stride=1, stage=None, bias_term=False):
        if self.check_stage(stage):
            self.increase_index()
            lr = self.check_lr_mult(lr)
            if weight_filler is None:
                weight_filler = dict(type = 'xavier')
            if self.phase == 'train' or self.phase == 'test':
                param = [dict(lr_mult = lr), dict(lr_mult = lr)] \
                    if bias_term else [dict(lr_mult = lr)]
                self.bottom = L.Convolution(self.bottom, param = param,
                    kernel_size = kernel_size, pad = pad,
                    stride = stride, num_output = num_output,
                    weight_filler = weight_filler,
                    bias_term = bias_term)
                    # bias_filler=dict(type='constant', value=0))
            elif self.phase == 'deploy':
                self.bottom = L.Convolution(self.bottom,
                    kernel_size = kernel_size, pad = pad,
                    stride = stride, num_output = num_output,
                    bias_term = bias_term)
            else:
                print("phase not supported")
            setattr(self.net, 'conv'+str(self.index), self.bottom)
        return self.bottom

    # deconvolutional layer
    def add_deconv(self, num_output, lr=1, kernel_size=2, weight_filler=None,
        pad=0, stride=2, stage=None, bias_term=False):
        if self.check_stage(stage):
            self.increase_index()
            lr = self.check_lr_mult(lr)
            if weight_filler is None:
                weight_filler = dict(type = 'xavier')
            if self.phase == 'train' or self.phase == 'test':
                param = [dict(lr_mult = lr), dict(lr_mult = lr)] \
                    if bias_term else [dict(lr_mult = lr)]
                self.bottom = L.Deconvolution(self.bottom, param = param,
                    convolution_param = dict(
                    kernel_size = kernel_size, pad = pad,
                    stride = stride, num_output = num_output,
                    weight_filler = weight_filler,
                     bias_term = bias_term,
                    bias_filler=dict(type='constant', value=0)))
            elif self.phase == 'deploy':
                self.bottom = L.Deconvolution(self.bottom,
                    convolution_param = dict(
                    kernel_size = kernel_size, pad = pad,
                    stride = stride, num_output = num_output,
                    bias_term = bias_term))
            setattr(self.net, 'deconv'+str(self.index), self.bottom)
        return self.bottom

    # 1d convolutional layer
    def add_conv_1d(self, num_output, lr=1, kernel_size=3, weight_filler=None,
        pad=1, stride=1, stage=None, bias_term=False):
        if self.check_stage(stage):
            self.increase_index()
            lr = self.check_lr_mult(lr)
            if weight_filler is None:
                weight_filler = dict(type = 'xavier')
            if self.phase == 'train' or self.phase == 'test':
                self.bottom = L.Convolution(self.bottom,
                    kernel_h=kernel_size, kernel_w=1,
                    pad_h=pad, pad_w=0, stride_h=stride, stride_w=1,
                    num_output=num_output, weight_filler=weight_filler,
                    param=dict(lr_mult=lr), bias_term=bias_term,
                    bias_filler=dict(type='constant', value=0))
            elif self.phase == 'deploy':
                self.bottom = L.Convolution(self.bottom,
                    kernel_h=kernel_size, kernel_w=1,
                    pad_h=pad, pad_w=0, stride_h=stride, stride_w=1,
                    num_output = num_output,
                    bias_term = bias_term)
            else:
                print("phase not supported")
            setattr(self.net, 'conv'+str(self.index), self.bottom)
        return self.bottom

    # add batch normalization
    def add_batchnorm(self, stage = None):
        if self.check_stage(stage):
            if self.phase == 'train':
                self.bottom = L.BatchNorm(self.bottom,
                    param = [dict(lr_mult = 0),
                             dict(lr_mult = 0), dict(lr_mult = 0)],
                    batch_norm_param = dict(use_global_stats = False),
                    in_place = True)
            elif self.phase == 'test':
                self.bottom = L.BatchNorm(self.bottom,
                    param = [dict(lr_mult = 0),
                             dict(lr_mult = 0), dict(lr_mult = 0)],
                    batch_norm_param = dict(use_global_stats = True),
                    in_place = True)
            elif self.phase == 'deploy':
                self.bottom = L.BatchNorm(self.bottom,
                    batch_norm_param = dict(use_global_stats = True),
                    in_place = True)
            setattr(self.net, 'bn'+str(self.index), self.bottom)
        return self.bottom

    # add scale function
    def add_scale(self, lr=1, stage=None):
        if self.check_stage(stage):
            lr = self.check_lr_mult(lr)
            if self.phase == 'train' or self.phase == 'test':
                self.bottom = L.Scale(self.bottom,
                      param = [dict(lr_mult = lr), dict(lr_mult = lr)],
                      bias_term = True, in_place = True,
                      bias_filler = dict(type = 'constant', value = 0))
            else:
                self.bottom = L.Scale(self.bottom,
                      param = [dict(lr_mult = lr), dict(lr_mult = lr)],
                      bias_term = True, in_place = True)
            setattr(self.net, 'scale'+str(self.index), self.bottom)
        return self.bottom

    # ReLU 
    def add_relu(self, stage = None):
        if self.check_stage(stage):
            self.bottom = L.ReLU(self.bottom, in_place = True)
            setattr(self.net, 'relu'+str(self.index), self.bottom)
        return self.bottom

    def add_dropout(self, dropout, stage=None):
        if self.check_stage(stage):
            self.bottom = L.Dropout(self.bottom, dropout_ratio = dropout,
                in_place = True)
            setattr(self.net, 'dropout'+str(self.index), self.bottom)
        return self.bottom

    # Parameterized ReLU 
    def add_prelu(self, lr = 1, stage = None):
        if self.check_stage(stage):
            lr = self.check_lr_mult(lr)
            self.bottom = L.PReLU(self.bottom, in_place = True, 
                  param = [dict(lr_mult = lr)])
            setattr(self.net, 'prelu'+str(self.index), self.bottom)
        return self.bottom

    # sinh layer
    def add_sinh(self, stage = None):
        if self.check_stage(stage):
            self.bottom = L.SinH(self.bottom, in_place = True)
            setattr(self.net, 'sinh'+str(self.index), self.bottom)
        return self.bottom

    # add scale function
    def add_const_scale(self, scale, stage=None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                self.bottom = L.Scale(self.bottom,
                      param = [dict(lr_mult=0)],
                      bias_term=False, in_place=True,
                      filler = dict(type='constant', value=scale))
            else:
                self.bottom = L.Scale(self.bottom,
                      param = [dict(lr_mult=0)],
                      bias_term=False, in_place=True)
            setattr(self.net, 'scale'+str(self.index), self.bottom)
        return self.bottom

    # add fc
    def add_fc(self, num_output, lr = 1, bias_value = 0,
            weight_filler=dict(type='xavier'), name = None, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            lr = self.check_lr_mult(lr)
            if not name:
                name = 'fc'+str(self.index)
            if self.phase == 'train' or self.phase == 'test':
                self.bottom = L.InnerProduct(self.bottom,
                    num_output = num_output,
                    param = [dict(lr_mult = lr), dict(lr_mult = lr)],
                    weight_filler=weight_filler,
                    bias_filler = dict(type = 'constant', value = bias_value))
            elif self.phase == 'deploy':
                self.bottom = L.InnerProduct(self.bottom,
                    num_output = num_output)
            setattr(self.net, name, self.bottom)
        return self.bottom

    def add_softmax_op(self, loss_weight = 1, stage = None):
        if self.check_stage(stage):
            name = 'softmax'+str(self.index)
            self.bottom = L.Softmax(self.bottom)
            setattr(self.net, name, self.bottom)
        return self.bottom

# cyclic functions
################################################################################
    def add_cslice(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.CyclicSlice(self.bottom)
            setattr(self.net, 'cslice'+str(self.index), self.bottom)
        return self.bottom

    def add_croll(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.CyclicRoll(self.bottom)
            setattr(self.net, 'croll'+str(self.index), self.bottom)
        return self.bottom

    def add_cpool(self, stage = None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.CyclicPool(self.bottom, pool = P.CyclicPool.AVE)
            setattr(self.net, 'cpool'+str(self.index), self.bottom)
        return self.bottom

# branch options
################################################################################
    def add_shortcut(self, stage=None):
        if self.check_stage(stage):
            self.shortcut_array.append(self.bottom)

    def concat_shortcut(self, stage=None):
        if self.check_stage(stage):
            self.increase_index()
            self.bottom = L.Concat(self.bottom, self.shortcut_array.pop())
            setattr(self.net, 'concat'+str(self.index), self.bottom)

# adversarial net
################################################################################
    def set_adversarial_discrimitive(self):
        self.adversarial_stage = 'discrimitive'
    def set_adversarial_generative(self):
        self.adversarial_stage = 'generative'

    # modify lr
    def check_lr_mult(self, lr_mult):
        if self.adversarial_stage == 'generative':
            lr_mult *= -1
        return lr_mult

    # mix adversarial and ground truth
    def mix_adversarial(self, G, R, stage = None):
        if self.check_stage(stage):
          if self.phase == 'train' or self.phase == 'test':
              self.bottom = L.Concat(G, R, axis = 0)
              setattr(self.net, 'adversarial_concat', self.bottom)
              dim = self.input_dim
              dim[2] = 1
              dim[3] = 1
              label0 = L.DummyData(shape = dict(dim = dim),\
                  data_filler = dict(type = 'constant', value = 0))
              label1 = L.DummyData(shape = dict(dim = dim),\
                  data_filler = dict(type = 'constant', value = 1))
              label = L.Concat(label0, label1, axis = 0)
              setattr(self.net, 'label_real', label0)
              setattr(self.net, 'label_gen', label1)
              setattr(self.net, 'adversarial_label', label)
              self.adversarial_label = label
        return self.bottom

    # pair up input and label
    def pair_data_label(self, data, label, stage = None):
        if self.check_stage(stage) and self.phase is not 'deploy':
            self.increase_index()
            pair = L.Concat(data, label, axis = 1)
            setattr(self.net, 'pair'+str(self.index), pair)
            return pair

    # convert label to softmax label
    def convert_label_softmax(self, num_classes, stage = None):
        if self.check_stage(stage) and self.phase is not 'deploy':
          self.label = L.LabelSoftmax(self.label, num_classes = num_classes)
          setattr(self.net, 'label_softmax', self.label)


# solvers
################################################################################
    # define sdg solver
    def add_solver_sdg(self, test_interval = 100, test_iter = 1,
                max_iter = 6e3, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-5, gamma = 0.1, stepsize = 2e3, iter_size=1,
                display = 10, snapshot = 1e3, fix_seed = False):
        solver = caffe_pb2.SolverParameter()
        if fix_seed:
            solver.random_seed = 0xCAFFE
        stage_str = str(len(self.solvers))
        solver.train_net = self.model_path+'train_'+stage_str+'.prototxt'
        solver.test_net.append(self.model_path+'test_'+stage_str+'.prototxt')
        solver.test_interval = int(test_interval)
        solver.test_iter.append(int(test_iter))
        solver.max_iter = int(max_iter)
        solver.iter_size = int(iter_size)
        solver.type = "SGD"
        solver.base_lr = base_lr
        solver.momentum = momentum
        solver.weight_decay = weight_decay
        solver.lr_policy = 'step'
        solver.gamma = gamma
        solver.stepsize  = int(stepsize)
        solver.display = int(display)
        solver.snapshot = int(snapshot)
        solver.snapshot_prefix = self.model_path+'stage_'+stage_str
        solver.solver_mode = caffe_pb2.SolverParameter.GPU
        self.solvers.append(solver)
        self.stage_iters.append(solver.max_iter)

# saving to files
################################################################################
    # write solver
    def save_solver(self):
        self.reset('train')
        for solver, stage in \
                zip(self.solvers, range(self.number_stages)):
            # training solver
            solver.test_initialization = False
            with open(self.model_path+'solver_'
                    +str(stage)+'.prototxt', 'w+') as f:
                f.write(str(solver))
            # pre-checking solver
            solver.test_initialization = True
            solver.max_iter = 1
            for test_iter_id in range(len(solver.test_iter)):
                solver.test_iter[test_iter_id] = 1
            with open(self.model_path+'solver_checking_'
                    +str(stage)+'.prototxt', 'w+') as f:
                f.write(str(solver))
            print(self.name+': '+'solver stage '+str(stage)+' writing finished')

    # save net
    def save_net(self):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        for phase in ['train', 'test', 'deploy']:
            for stage in range(self.number_stages):
                self.reset(phase, stage)
                with open(self.model_path+self.phase+'_'
                        +str(stage)+'.prototxt', 'w+') as f:
                    f.write('name: "'+self.name+'"\n')
                    f.write(str(self.net.to_proto()))
                    print(self.name+': stage '+str(stage)+' '+ \
                            self.phase+' net writing finished!')

    # generate runfile header
    def gen_runfile_header(self, f):
        f.write('GPU=0\n'+
                'REPEAT=1\n'
                'while [[ $# -gt 1 ]]\n'+
                'do\n'+
                'key="$1"\n'+
                'case $key in\n'+
                '-g|--gpu)\n'+
                'GPU="$2"\n'+
                'shift # past argument\n'+
                ';;\n'+
                '-r|--repeat)\n'+
                'REPEAT="$2"\n'+
                'shift # past argument\n'+
                ';;\n'+
                '*)\n'+
                '# unknown option\n'
                ';;\n'+
                'esac\n'+
                'shift # past argument\n'+
                'done\n')

    # save runfile
    def save_runfile(self, weights):
        with open(self.model_path+'runfile.sh', 'w+') as f:
            self.gen_runfile_header(f)
            self.reset()
            for stage in range(self.number_stages):
                f.write(self.caffe_path+'build/tools/caffe train -gpu $GPU'+
                        ' \\\n--solver=models/'+self.name+
                        '/solver_'+str(stage)+'.prototxt')
                if stage > 0:
                    f.write(' \\\n--weights=models/'+self.name+
                            '/stage_'+str(stage-1)+'_iter_'+
                            str(self.stage_iters[stage-1])+'.caffemodel')
                if stage == 0 and weights is not None:
                    f.write(' \\\n--weights='+weights)
                f.write(' \\\n2>&1 | tee ')
                if stage > 0:
                    f.write('-a ')
                f.write('models/'+self.name+'/log_$REPEAT.txt\n')
                f.write('cp models/'+self.name+'/stage_'+str(stage)+'_iter_'+
                        str(self.stage_iters[stage])+'.caffemodel \\\n'+
                        'models/'+self.name+'/stage_'+str(stage)+
                        '_final_$REPEAT.caffemodel\n')

    # save runfile
    def save_checking(self):
        with open(self.model_path+'checking.sh', 'w+') as f:
            self.gen_runfile_header(f)
            self.reset()
            f.write('set -e\n')
            for stage in range(self.number_stages):
                f.write(self.caffe_path+'build/tools/caffe train -gpu $GPU'+
                        ' \\\n--solver=models/'+self.name+
                        '/solver_checking_'+str(stage)+'.prototxt \\\n')
                if stage == 0:
                    f.write('2>&1 | tee ')
                else:
                    f.write('2>&1 | tee -a ')
                f.write('models/'+self.name+'/log_checking.txt\n')
                f.write('rm models/'+self.name+'/stage_'+str(stage)+
                        '_iter_1.*\n')
            f.write('set +e\n')

    # save
    def save(self, pretrained_weights = None):
        self.save_net()
        self.save_solver()
        self.save_runfile(pretrained_weights)
        self.save_checking()