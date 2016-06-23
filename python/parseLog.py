# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 00:41:10 2016
result is a dictionary that have different keys including, 'train', 
'train_name', 'train_name_weight', 'train_name_weighted', 'lr',
'test[i]', 'test[i]_name'
@author: yz
"""
import re

def addValue(dic, keyname, iteration, value):
    if not dic.has_key(keyname):
        dic[keyname] = {iteration:value}
    else:
        dic[keyname][iteration]=value
        
def parseLog(filename):
    result = {};
    with open(filename,'r') as file:
        for line in file:
            #print(line)
            # match iteration number for loss
            iterMatch = re.match('.* Iteration (\d*).*',line)
            if iterMatch:
                iteration = int(iterMatch.group(1))
            trainlossMatch = re.match('.* Iteration (\d*), loss = (\d*\.?\d*)',line)
            if trainlossMatch:
                iteration = int(trainlossMatch.group(1))
                totalTrainLoss = float(trainlossMatch.group(2))
                addValue(result, 'train', iteration, totalTrainLoss)
            # match test net id
            testIDMatch = re.match('.* Testing net \(#(\d*)\)',line)
            if testIDMatch:
                testID = (testIDMatch.group(1))
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
                addValue(result, phase+'_'+name, iteration, value)
                if outputMatch.group(5)=='':
                    weight = float('nan')
                    weighted_value = float('nan')
                else:
                    weight = float(outputMatch.group(5))
                    weighted_value = float(outputMatch.group(6))
                    addValue(result, phase+'_weight', iteration, weight)
                    addValue(result, phase+'_weighted', iteration, weighted_value)
            # match lear rate
            lrMatch = re.match('.* lr = (\d*\.?\d*).*',line)
            if lrMatch:
                lr = float(lrMatch.group(1))
                addValue(result, 'lr', iteration, weighted_value)
    return result

