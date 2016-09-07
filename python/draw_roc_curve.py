# predicted result
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

parser = argparse.ArgumentParser(description =
    'Generate the ROC curve based on predicted result')
parser.add_argument('--INPUT', type = str, default = None)
parser.add_argument('--OUTPUT', type = str, default = None)

parsed = parser.parse_args()
inputfile = parsed.INPUT
outputfile = parsed.OUTPUT

prob = []
label = []
with open(inputfile, 'rb') as csvfile:
    csvreader = csv.reader(csvfile, delimiter = ' ')
    for row in csvreader:
        prob.append(float(row[3]))
        label.append(float(row[1]))
        
prob = np.asanyarray(prob)
label = np.asanyarray(label)

fpr, tpr, _ = roc_curve(label, prob)
roc_auc = auc(fpr, tpr)
    
f=plt.figure(figsize = (20,10))

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('false postive rate')
plt.ylabel('true positive rate')
plt.legend(loc="lower right")
plt.savefig(outputfile)
