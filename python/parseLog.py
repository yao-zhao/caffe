# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:41:10 2016
result is a dictionary that have different keys including, 'train', 
'train_name', 'train_name_weight', 'train_name_weighted', 'lr',
'test[i]', 'test[i]_name'
@author: yz
"""
import re
def addValue(dic, keyname, subkeyname, value):
    if not dic.has_key(keyname):
        dic[keyname] = {}
    if not dic[keyname].has_key(subkeyname):
        dic[keyname][subkeyname]=[]
    dic[keyname][subkeyname].append(value)
        
def parseLog(filename):
    result = {};
    with open(filename,'r') as file:
        for line in file:
            #print(line)
            # match iteration number for loss
            testIDMatch = re.match('.* Iteration (\d*), Testing net \(#(\d*)\).*',line)
            if testIDMatch:
                iteration = int(testIDMatch.group(1))
                testID = (testIDMatch.group(2))
                addValue(result, 'test'+testID,'iter',iteration)

            trainlossMatch = re.match('.* Iteration (\d*), loss = (\d*\.?\d*)',line)
            if trainlossMatch:
                iteration = int(trainlossMatch.group(1))
                totalTrainLoss = float(trainlossMatch.group(2))
                addValue(result, 'train', 'iter',iteration)
                addValue(result, 'train', 'total_loss',totalTrainLoss)
            # match test net id
#            testIDMatch = re.match('.* Testing net \(#(\d*)\)',line)
#            if testIDMatch:
            # match output
            outputMatch = re.match(
            '.* (.*) net output #?(\d*)?: (\S*)'+
            ' = (\d*.?\d*)(?: \(\* )?(\d*\.?\d*)?(?: = )?(\d*\.?\d*)?', line)
            if outputMatch:
                phase = outputMatch.group(1).lower()
                index= int(outputMatch.group(2))
                name = outputMatch.group(3)
                value = float(outputMatch.group(4))
                if phase == 'test':
                    phase += testID
                addValue(result, phase, name, value)
                if outputMatch.group(5)=='':
                    weight = float('nan')
                    weighted_value = float('nan')
                else:
                    weight = float(outputMatch.group(5))
                    weighted_value = float(outputMatch.group(6))
                    addValue(result, phase, name+'_weight', weight)
                    addValue(result, phase, name+'_weighted', weighted_value)
            # match lear rate
            lrMatch = re.match('.* lr = (\d*\.?\d*).*',line)
            if lrMatch:
                lr = float(lrMatch.group(1))
                addValue(result, 'train', 'lr', lr)
                
    return result
