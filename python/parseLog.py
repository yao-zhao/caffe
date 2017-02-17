# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:41:10 2016
result is a dictionary that have different keys including, 'train', 
'train_name', 'train_name_weight', 'train_name_weighted', 'lr',
'test[i]', 'test[i]_name'
@author: yz
"""
import re

def parseLog(filename):
    # first iteration number of each stage
    base_iteration = 0
    last_iteration = 0
    iteration = 0
    def addValue(dic, keyname, subkeyname, i, value):
        if keyname not in dic.keys():
            dic[keyname] = {}
        if subkeyname not in dic[keyname]:
            dic[keyname][subkeyname] = []
        dic[keyname][subkeyname].append((i+base_iteration, value))    
    # start parsing
    result = {};
    with open(filename,'r') as file:
        for line in file:
            # match iteration number, testID
            testIDMatch = re.match(
                '.* Iteration (\d*), Testing net \(#(\d*)\).*',line)
            if testIDMatch:
                iteration = int(testIDMatch.group(1))
                testID = (testIDMatch.group(2))
#                addValue(result, 'test'+testID,'iter',iteration)
            # match train total loss
            trainlossMatch = re.match(
                '.* Iteration (\d*), loss = ([-]?\d*\.?\d*)',line)
            if trainlossMatch:
                iteration = int(trainlossMatch.group(1))
                totalTrainLoss = float(trainlossMatch.group(2))
#                addValue(result, 'train', 'iter', iteration)
                addValue(result, 'train', 'total_loss', 
                         iteration, totalTrainLoss)
            # match output
            outputMatch = re.match(
            '.* (.*) net output #?(\d*)?: (\S*)'+
            ' = ([-]?\d*.?\d*)(?: \(\* )?([-]?\d*\.?\d*)?(?: = )?'+
            '([-]?\d*\.?\d*)?', line)
            if outputMatch:
                phase = outputMatch.group(1).lower()
                index = int(outputMatch.group(2))
                name = outputMatch.group(3)
                value = float(outputMatch.group(4))
                if phase == 'test':
                    phase += testID
                addValue(result, phase, name, iteration, value)
                if outputMatch.group(5) =='':
                    weight = float('nan')
                    weighted_value = float('nan')
                else:
                    weight = float(outputMatch.group(5))
                    weighted_value = float(outputMatch.group(6))
                    addValue(result, phase, name+'_weight',
                             iteration, weight)
                    addValue(result, phase, name+'_weighted',
                             iteration, weighted_value)
            # match lear rate
            lrMatch = re.match('.* lr = ([-]?\d*\.?\d*).*',line)
            if lrMatch:
                lr = float(lrMatch.group(1))
                addValue(result, 'train', 'lr', iteration, lr)
            if last_iteration > iteration:
                base_iteration += last_iteration
                last_iteration = iteration
            elif last_iteration < iteration:
                last_iteration = iteration
#    result['train']['total_loss'] = result['train']['total_loss'][1:]
#    result['train']['iter'] = result['train']['iter'][1:]
    return result
