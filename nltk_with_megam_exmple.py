# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 14:13:04 2018

@author: Eric
"""
import random
from nltk.corpus import names
from nltk import MaxentClassifier
#from nltk import classify
import nltk
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
random.shuffle(names)

def gender_features3(name):
    features = {}
    features["fl"] = name[0].lower()
    features["ll"] = name[-1].lower()
    features["fw"] = name[:2].lower()
    features["lw"] = name[-2:].lower()
    return features

featuresets = [(gender_features3(n), g) for (n, g) in names]

train_set, test_set = featuresets[500:], featuresets[:500]
#get answer from https://groups.google.com/forum/#!topic/nltk-users/X2nld28FYBw
nltk.config_megam('/home/xiao/Downloads/megam_0.92')# it works!!!


me3_megam_classifier = MaxentClassifier.train(train_set, "megam")