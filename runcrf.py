# -*- coding: utf8 -*-
import sys
import glob
import random
import pycrfsuite
import crf
import util
import datetime

material = 'data/sjw/*'
#material = "data/sjw/A05*"
size = 80
trainportion = 0.9
cut = int(size*trainportion)
dictfile = 'data/vector/vectors300.txt'
crfmethod = "l2sgd"  # {‘lbfgs’, ‘l2sgd’, ‘ap’, ‘pa’, ‘arow’}
charstop = True # True means label attributes to previous char
features = 3 # 1=discrete; 2=vectors; 3=both
random.seed(101)

"python runcrf.py 'data/sjw/*' 80 data/vector/vectors300.txt 1 1"
args = sys.argv
if len(args)>1:
    material = args[1]
    size = int(args[2])
    dictfile = args[3]
    features = int(args[4])
    charstop = int(args[5])


modelname = material.replace('/','').replace('*','')+str(size)+".m"

print "Material:", material
print "Size:", size, "entries,", trainportion, "as training"

starttime = datetime.datetime.now()
print "Starting Time:",starttime
# Prepare li: list of random lines
if features > 1: vdict = util.readvec(dictfile)
li = [line for line in util.file_to_lines(glob.glob(material))]
random.shuffle(li)
li = li[:size]

# Prepare data: list of x(char), y(label) sequences
data = []
for line in li:
    x, y = util.line_toseq(line, charstop)
    if features == 1:
        d = crf.x_seq_to_features_discrete(x, charstop), y
    elif features == 2:
        d = crf.x_seq_to_features_vector(x, vdict, charstop), y
    elif features == 3:
        d = crf.x_seq_to_features_both(x, vdict, charstop), y
    data.append(d)

traindata = data[:cut]
testdata = data[cut:]

trainer = pycrfsuite.Trainer()
#print trainer.params()

for t in traindata:
    x, y = t
    trainer.append(x, y)

trainer.select(crfmethod)
trainer.train(modelname)

tagger = pycrfsuite.Tagger()
tagger.open(modelname)
tagger.dump(modelname+".txt")

print "Start testing..."
results = []
for t in testdata:
    x, yref = t
    yout = tagger.tag(x)
    results.append(util.eval(yref, yout, "S"))

tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Test Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "*******************F1-score:", 2*p*r/(p+r)



print "Start closed testing..."
results = []
for t in traindata:
    x, yref = t
    yout = tagger.tag(x)
    results.append(util.eval(yref, yout, "S"))

tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Train Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "*******************F1-score:", 2*p*r/(p+r)
