# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 16:14:14 2016
visualize logs
@author: yz
"""
import matplotlib
import matplotlib.pyplot as plt
import parseLog
import argparse
import numpy as np

parser = argparse.ArgumentParser(description =
    'Draw training comparision between current and basic')
parser.add_argument('--CURRENT_LOG', type = str, default = None)
parser.add_argument('--BASE_LOG', type = str, default = None)
parser.add_argument('--OUTPUT', type = str, default = 'train.png')
parser.add_argument('--LOSS_NAME', type = str, default = 'loss')
parser.add_argument('--Y_MAX', type = float, default = None)
parser.add_argument('--Y_MIN', type = float, default = None)

parsed = parser.parse_args()
log_current = parsed.CURRENT_LOG
log_base = parsed.BASE_LOG
destination = parsed.OUTPUT
lossname = parsed.LOSS_NAME
ymax = parsed.Y_MAX
ymin = parsed.Y_MIN

# seperate loss names
lossname = my_string = [x.strip() for x in lossname.split(',')]

# get iter
def getiter(l):
    return [x[0] for x in l]
def getvalue(l):
    return [x[1] for x in l]
    
f=plt.figure(figsize = (20,10))

legends=[]
totalmin1 = 100000
totalmax1 = -100000
itersmax1 = -100000
itersmin1 = 100000
# plot
def plot(log):
    global totalmin1, totalmax1, itersmax1, itersmin1
    result1 = parseLog.parseLog(log)
    train1 = result1['train']
    test1 = result1['test0']
    for ln in lossname:
        plt.plot(getiter(train1[ln]),getvalue(train1[ln]))
        legends.append(log+' '+ln+' '+'train')
        plt.plot(getiter(test1[ln]),getvalue(test1[ln]))
        legends.append(log+' '+ln+' '+'val')
        totalmin1 = min(totalmin1, np.min(np.concatenate(
            (getvalue(train1[ln]), getvalue(test1[ln])), axis=0)))
        totalmax1 = max(totalmax1, np.mean(np.concatenate(
            (getvalue(train1[ln]), getvalue(test1[ln])), axis=0)))
    itersmax1 = max(itersmax1, np.max(getiter(train1[ln])))
    itersmin1 = min(itersmin1, np.min(getiter(train1[ln])))

if log_current:
    plot(log_current)
if log_base:
    plot(log_base)

if not ymax:
  ymax = totalmax1+0.1
if not ymin:
  ymin = totalmin1
plt.axis([itersmin1, itersmax1, ymin, ymax])
plt.legend(legends, loc = 'lower left')
plt.savefig(destination)
