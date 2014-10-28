# -*- coding: utf8 -*-
import sys
import glob
import random
import pycrfsuite
import crf
import util
import datetime

material = 'data/sjw/*'
dictfile = 'data/vector/vectors300.txt'
charstop = True # True means label attributes to previous char
features = 3 # 1=discrete; 2=vectors; 3=both
random.seed(101)

"python runcrf-hu.py 'qualitative/allover-sjw-gold.*' d 1 1 datasjw1000001.m"
"python runcrf-hu.py 'qualitative/allover-24s-gold.*' d 1 1 data24s100001.m"
args = sys.argv
if len(args)>1:
    material = args[1]
    dictfile = args[2]
    features = int(args[3])
    charstop = int(args[4])
    modelname = args[5]

print "Material:", material

print datetime.datetime.now()

# Prepare li: list of random lines
if features > 1:
    vdict = util.readvec(dictfile)
    print "Dict:", dictfile
li = [line for line in util.file_to_lines(glob.glob(material))]

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

tagger = pycrfsuite.Tagger()
tagger.open(modelname)

print datetime.datetime.now()
print "Start testing..."
results = []
lines = []
while data:
    xseq, yref = data.pop()
    yout = tagger.tag(xseq)
    results.append(util.eval(yref, yout, "S"))
    lines.append(util.seq_to_line([x['gs0'] for x in xseq],yout,charstop))

tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Test Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "*******************F1-score:", 2*p*r/(p+r)

for line in lines:
    print line.encode('utf8')
