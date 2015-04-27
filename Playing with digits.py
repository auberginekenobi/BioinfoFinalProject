# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 20:02:06 2015

@author: chapmano
"""

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
digits = datasets.load_digits()
import pylab as pl

print (digits.data.shape)
#(1797,64) (number of samples, dimensionality of each sample)

# train the classifier on all but the last 100 sample of the digits set
classifier = RandomForestClassifier()
classifier = classifier.fit(digits.data[:-100],digits.target[:-100])

# check on the last sample in the digits set
pl.gray()
pl.matshow(digits.images[-1])
pl.show()
print "classified as",classifier.predict(digits.data[-1])
print digits.target[-1]

#check number 2 on the last 100 samples
i=1
c=0.0
while i<=100:
    if classifier.predict(digits.data[-i])==digits.target[-i]:
        c+=1
    i+=1
print "classifier accuracy:", c/i
