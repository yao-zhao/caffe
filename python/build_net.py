# -*- coding: utf-8 -*-
"""
collection of quick helper to build layers
"""

import os
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import numpy as np

class BuildNet:
    # ini def
    def __init__(self, func, name = 'net', savepath = 'models/'):
        self.func = func
        self.model_path = savepath + name +'/'
        self.name = name
        self.reset()

# common task
################################################################################
    # increase index
    def increase_index(self):
        self.index += 1

    # change the bottom
    def set_bottom(self, name):
        if type(name) in [str, unicode]:
            self.bottom = getattr(self.net, name)
        else:
            self.bottom = name

    # reset net
    def reset(self):
        self.index = 1
        self.net = None
        self.net = caffe.NetSpec()
        self.solvers = []
        self.stage = None
        self.stage_iters = []
        self.bottom = None
        self.label = None
        self.phase = 'train'

    def number_stages(self):
        return len(self.solvers)

    # check stage
    def check_stage(self, stage):
        return not stage or not self.stage or \
                np.any(np.equal(stage, self.stage))

# function groups
################################################################################
    # add a typical block
    def add_normal_block(self, num_output, lr = 1):
        self.add_conv(num_output, lr = lr)
        self.add_batchnorm()
        self.add_scale(lr = lr)
        self.add_relu()
        self.index += 1

    # add a prelu block
    def add_prelu_block(self, num_output, lr = 1):
        self.add_conv(num_output, lr = lr)
        self.add_batchnorm()
        self.add_scale(lr = lr)
        self.add_prelu()
        self.index += 1

    # add a conv pool block
    def add_convpool_block(self, num_output, lr = 1):
        self.add_conv(num_output, lr = lr, stride = 2)
        self.add_batchnorm()
        self.add_scale(lr = lr)
        self.add_relu()
        self.index += 1

    # add a prelu conv pool block
    def add_prelu_convpool_block(self, num_output, lr = 1):
        self.add_conv(num_output, lr = lr, stride = 2)
        self.add_batchnorm()
        self.add_scale(lr = lr)
        self.add_prelu()
        self.index += 1

# input layers
################################################################################
    # set data layer
    def add_lmdb(self, transformer_dict = None, batch_size = 32,
                 test_batch_size = None,
                 backend = P.Data.LMDB, source_path = 'data/'):
        if test_batch_size is None:
            test_batch_size = batch_size
        if self.phase == 'train':
            self.bottom, self.label = L.Data(batch_size = batch_size, 
                    backend = backend, source = source_path + 'train_lmdb',
                    transform_param = transformer_dict,
                    ntop = 2, include = dict(phase = caffe.TRAIN))
            self.net.data = self.bottom
            self.net.label = self.label
        elif self.phase == 'test':
            self.bottom, self.label = L.Data(batch_size = test_batch_size,
                    backend = backend, source = source_path + 'val_lmdb',
                    transform_param = transformer_dict,
                    ntop = 2, include = dict(phase = caffe.TEST))
            self.net.data = self.bottom
            self.net.label = self.label
        elif self.phase == 'deploy':
            self.net.data = self.bottom
        return self.bottom

    # set image data layer
    def add_image(self, transformer_dict = None, batch_size = 32,
                 test_batch_size = None,
                 source_path = 'data/', root_folder = 'data/',
                 shuffle = True, is_color = True, height = None, width = None):
        if test_batch_size is None:
            test_batch_size = batch_size
        # probe image data dimension
        if not height or not width:
            tmpnet = caffe.NetSpec()
            tmpnet.data, tmpnet.label = L.ImageData(
                source = source_path+'train.txt',
                root_folder = root_folder, is_color = is_color,
                batch_size = 1, ntop = 2)
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
                    shuffle = shuffle,
                    transform_param = transformer_dict, ntop = 2,
                    new_height = height, new_width = width,
                    include = dict(phase = caffe.TRAIN))
            self.net.data = self.bottom
            self.net.label = self.label
        elif self.phase == 'test':
            self.bottom, self.label = L.ImageData(batch_size = test_batch_size,
                    source = source_path + 'val.txt',
                    root_folder = root_folder, is_color = is_color,
                    shuffle = shuffle,
                    transform_param = transformer_dict, ntop = 2,
                    new_height = height, new_width = width,
                    include = dict(phase = caffe.TEST))
            self.net.data = self.bottom
            self.net.label = self.label
        elif self.phase == 'deploy':
            self.net.data = self.bottom
        return self.bottom

# pooling layers
################################################################################
    # add pooling layer of 2
    def add_maxpool_2(self):
        self.bottom = L.Pooling(self.bottom,
            kernel_size = 2, stride = 2, pool = P.Pooling.MAX)
        setattr(self.net, 'pool'+str(self.index), self.bottom)
        self.index += 1
        return self.bottom

    # add final mean pool
    def add_meanpool_final(self):
        self.bottom = L.Pooling(self.bottom, global_pooling = True,
            stride = 1, pool = P.Pooling.AVE)
        setattr(self.net, 'pool'+str(self.index), self.bottom)
        self.index += 1
        return self.bottom

# output layers
################################################################################
    def add_softmax(self, loss_weight = 1, name = 'softmax', stage = None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                softmax = L.SoftmaxWithLoss(self.bottom, self.label,
                                            loss_weight = loss_weight)
                accuracy = L.Accuracy(self.bottom, self.label)
                setattr(self.net, name+'loss', softmax)
                setattr(self.net, 'accuracy', accuracy)
            elif self.phase == 'deploy':
                softmax = L.Softmax(self.bottom)
                setattr(self.net, name+'prob', softmax)

    # add softmax
    def add_euclidean(self, loss_weight = 1, name = 'euclidean', stage = None):
        if self.check_stage(stage):
            if self.phase == 'train' or self.phase == 'test':
                euclidean = L.EuclideanLoss(self.bottom, self.label,
                                          loss_weight = loss_weight)
                setattr(self.net, name+'loss', euclidean)

    # add gaussian prob loss layer
    def add_gaussian_prob(self, mean, var, label = None,
        eps = 1e-2, loss_weight = 1, name = 'gaussianprob', stage = None):
        if self.check_stage(stage):
            if not label:
                label = self.label
            if self.phase == 'train' or self.phase == 'test':
                gaussianprob = L.GaussianProbLoss(mean, var, label,
                    gaussian_prob_loss_param = dict(eps = eps),
                    loss_weight = loss_weight)
                setattr(self.net, name+'loss', gaussianprob)

    # depreciated function
    # # add gaussian prob loss layer with fc
    # def add_gaussian_prob(self, mean_lr = 1, var_lr = 0.1, eps = 1e-2,
    #     loss_weight = 1, name = 'gaussianprob',
    #     var_bias = 1):
    #     if self.phase == 'train' or self.phase == 'test':
    #         mean = L.InnerProduct(self.bottom, num_output = 1,
    #             param = [dict(lr_mult = mean_lr), dict(lr_mult = mean_lr)],
    #             weight_filler = dict(type = 'xavier'),
    #             bias_filler = dict(type = 'constant', value = 0))
    #         var = L.InnerProduct(self.bottom, num_output = 1,
    #             param = [dict(lr_mult = var_lr), dict(lr_mult = var_lr)],
    #             weight_filler = dict(type = 'xavier'),
    #             bias_filler = dict(type = 'constant', value = var_bias))
    #         relu = L.ReLU(var, in_place = True)
    #         setattr(self.net, 'fcmean'+str(self.index), mean)
    #         setattr(self.net, 'fcvar'+str(self.index), var)
    #         setattr(self.net, 'relu'+str(self.index), relu)
    #         self.index += 1
    #         gaussianprob = L.GaussianProbLoss(mean, var, self.label,
    #             gaussian_prob_loss_param = dict(eps = eps),
    #             loss_weight = loss_weight)
    #         setattr(self.net, name+'loss', gaussianprob)
    #         self.bottom = mean
    #     elif self.phase == 'deploy':
    #         mean = L.InnerProduct(self.bottom, num_output = 1)
    #         var = L.InnerProduct(self.bottom, num_output = 1)
    #         relu = L.ReLU(var, in_place = True)
    #         setattr(self.net, 'fcmean'+str(self.index), mean)
    #         setattr(self.net, 'fcvar'+str(self.index), var)
    #         setattr(self.net, 'relu'+str(self.index), relu)
    #         self.index += 1

# common building components
################################################################################
    # convolutional layer 
    def add_conv(self, num_output, lr = 1, kernel_size = 3,
        pad = 1, stride = 1):
        if self.phase == 'train' or self.phase == 'test':
            self.bottom = L.Convolution(self.bottom,
                kernel_size = kernel_size, pad = pad,
                stride = stride, num_output = num_output,
                weight_filler = dict(type = 'xavier'),
                param = dict(lr_mult = lr), bias_term = False)
        elif self.phase == 'deploy':
            self.bottom = L.Convolution(self.bottom,
                kernel_size = kernel_size, pad = pad,
                stride = stride, num_output = num_output,
                bias_term = False)
        else:
            print "phase not supported"
        setattr(self.net, 'conv'+str(self.index), self.bottom)
        return self.bottom

    # add batch normalization
    def add_batchnorm(self):
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
    def add_scale(self, lr = 1):
        if self.phase == 'train' or self.phase == 'test':
            self.bottom = L.Scale(self.bottom,
                  param = [dict(lr_mult = lr), dict(lr_mult = lr)],
                  bias_term = True, in_place = True,
                  bias_filler = dict(type = 'constant',value = 0))
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

    # Parameterized ReLU 
    def add_prelu(self, lr = 1):
        self.bottom = L.PReLU(self.bottom, in_place = True, 
              param = [dict(lr_mult = lr)])
        setattr(self.net, 'prelu'+str(self.index), self.bottom)
        return self.bottom

    # add fc
    def add_fc(self, num_output, lr = 1, dropout = 0, bias_value = 0,
            weight_filler = 'xavier', name = None, stage = None):
        if self.check_stage(stage):
            if not name:
                name = 'fc'+str(self.index)
            if self.phase == 'train' or self.phase == 'test':
                self.bottom = L.InnerProduct(self.bottom,
                    num_output = num_output,
                    param = [dict(lr_mult = lr), dict(lr_mult = lr)],
                    weight_filler = dict(type = weight_filler),
                    bias_filler = dict(type = 'constant', value = bias_value))
            elif self.phase == 'deploy':
                self.bottom = L.InnerProduct(self.bottom,
                    num_output = num_output)
            setattr(self.net, name, self.bottom)
            if dropout > 0:
                self.bottom = L.Dropout(self.bottom, dropout_ratio = dropout,
                    in_place = True)
                setattr(self.net, 'dropout'+str(self.index), self.bottom)
        self.index += 1
        return self.bottom

# cyclic functions
################################################################################
    def add_cslice(self):
        self.bottom = L.CyclicSlice(self.bottom)
        setattr(self.net, 'cslice', self.bottom)
        self.index += 1
        return self.bottom

    def add_croll(self):
        self.bottom = L.CyclicRoll(self.bottom)
        setattr(self.net, 'croll'+str(self.index), self.bottom)
        self.index += 1
        return self.bottom

    def add_cpool(self):
        self.bottom = L.CyclicPool(self.bottom, pool = P.CyclicPool.AVE)
        setattr(self.net, 'cpool', self.bottom)
        self.index += 1
        return self.bottom

# solvers
################################################################################
    # define sdg solver
    def add_solver_sdg(self, test_interval = 100, test_iter = 1,
                max_iter = 6e3, base_lr = 0.01, momentum = 0.9,
                weight_decay = 1e-5, gamma = 0.1, stepsize = 2e3,
                display = 10, snapshot = 1e3):
        solver = caffe_pb2.SolverParameter()
        # solver.random_seed = 0xCAFFE
        solver.train_net = self.model_path+'train_'+\
                str(self.number_stages())+'.prototxt'
        solver.test_net.append(self.model_path+'test_'+\
                str(self.number_stages())+'.prototxt')
        solver.test_interval = test_interval
        solver.test_iter.append(test_iter)
        solver.max_iter = int(max_iter)
        solver.type = "SGD"
        solver.base_lr = base_lr
        solver.momentum = momentum
        solver.weight_decay = weight_decay
        solver.lr_policy = 'step'
        solver.gamma = gamma
        solver.stepsize  = int(stepsize)
        solver.display = int(display)
        solver.snapshot = int(snapshot)
        solver.snapshot_prefix = self.model_path+'stage_'+\
                str(self.number_stages())
        solver.solver_mode = caffe_pb2.SolverParameter.GPU
        self.solvers.append(solver)
        self.stage_iters.append(solver.max_iter)

# saving to files
################################################################################
    # write solver
    def save_solver(self):
        self.reset()
        self.func(self)
        for solver, stage in \
                zip(self.solvers, range(self.number_stages())):
            # training solver
            solver.test_initialization = False
            with open(self.model_path+'solver_'
                    +str(stage)+'.prototxt', 'w+') as f:
                f.write(str(solver))
            # pre-checking solver
            solver.test_initialization = True
            solver.max_iter = 1
            with open(self.model_path+'solver_checking_'
                    +str(stage)+'.prototxt', 'w+') as f:
                f.write(str(solver))
            print self.name+': '+'solver stage '+str(stage)+' writing finished'

    # save net
    def save_net(self):
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        for phase in ['train','test','deploy']:
            self.reset()
            self.phase = phase
            self.func(self)
            for stage in range(self.number_stages()):
                self.stage  = stage
                with open(self.model_path+self.phase+'_'
                        +str(stage)+'.prototxt', 'w+') as f:
                    f.write('name: \"'+self.name+'\"\n')
                    if phase == 'deploy':
                        numbatch, numchannel, height, width = caffe.Net(
                            self.model_path+'train_0.prototxt',
                            caffe.TRAIN).blobs['data'].shape
                        f.write('input: \"data\"\n')
                        f.write('input_dim: '+str(numbatch)+'\n')
                        f.write('input_dim: '+str(numchannel)+'\n')
                        f.write('input_dim: '+str(height)+'\n')
                        f.write('input_dim: '+str(width)+'\n')
                    f.write(str(self.net.to_proto() ))
                    print self.name+': stage '+str(stage)+' '+ \
                            self.phase+' net writing finished!'

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
                '# unknown option'
                ';;\n'+
                'esac\n'+
                'shift # past argument\n'+
                'done\n')

    # save runfile
    def save_runfile(self):
        with open(self.model_path+'runfile.sh', 'w+') as f:
            self.gen_runfile_header(f)
            self.reset()
            self.func(self)
            for stage in range(self.number_stages()):
                f.write('~/caffe-yao/build/tools/caffe train -gpu $GPU'+
                        ' \\\n--solver=models/'+self.name+
                        '/solver_'+str(stage)+'.prototxt')
                if stage > 0:
                    f.write(' \\\n--weights=models/'+self.name+
                            '/stage_'+str(stage)+'_iter_'+
                            str(self.stage_iters[stage-1])+'.solverstate')
                f.write(' \\\n2>&1 | tee ')
                if stage > 0:
                    f.write('-a ')
                f.write('models/'+self.name+'/log_$REPEAT.txt\n')

    # save runfile
    def save_checking(self):
        with open(self.model_path+'checking.sh', 'w+') as f:
            self.gen_runfile_header(f)
            self.reset()
            self.func(self)
            for stage in range(self.number_stages()):
                f.write('~/caffe-yao/build/tools/caffe train -gpu $GPU'+
                        ' \\\n--solver=models/'+self.name+
                        '/solver_checking_'+str(stage)+'.prototxt')
                f.write(' \\\n2>&1 | tee ')
                if stage > 0:
                    f.write('-a ')
                f.write('models/'+self.name+'/log_checking_$REPEAT.txt\n')

    # save
    def save(self):
        # self.save_net()
        self.save_solver()
        self.save_runfile()
        self.save_checking()