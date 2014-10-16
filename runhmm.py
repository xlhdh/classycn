# -*- coding: utf8 -*-
import sys
import glob
import datetime
import random
import nltk.tag as nt
import util

" PARAMETERS "
material = 'data/sjw/*'
#material = "data/sjw/A05*"
size = 80
trainportion = 0.9
charstop = False # True means label attributes to previous char
random.seed(101)
" END OF PARAMETERS "

"python runhmm.py 'data/sjw/*' 80 1"
args = sys.argv
if len(args)>1:
    material = args[1]
    size = int(args[2])
    charstop = int(args[5])
cut = int(size*trainportion)

print "Material:", material
print "Size:", size, "entries,", trainportion, "as training"

print "Starting Time:",datetime.datetime.now()

# Prepare li: list of random lines
print "Reading from files..."
li = [line for line in util.file_to_lines(glob.glob(material))]
random.shuffle(li)
li = li[:size]

# Prepare data: list of x(char), y(label) sequences
print "Prepare list of sequences..."

closetestdata = li[:cut]
testdata = li[cut:]

traindata = []
for line in closetestdata:
    x, y = util.line_toseq(line, charstop)
    traindata.append(zip(x,y))

# traindata shape: [[(x,y),(x,y), ...],[],[],...]
# testdata shape: [([x1, x2, ...],[y1,y2,...]),([],[])]

stt = datetime.datetime.now()
print "Start training...", stt
hmmtagger = nt.hmm.HiddenMarkovModelTagger.train(traindata)


print "################# Training took:", datetime.datetime.now()-stt
results = []
for line in testdata:
    x, yref = util.line_toseq(line, charstop)
    out = hmmtagger.tag(x)
    _, yout = zip(*out)
    results.append(util.eval(yref, yout, "S"))
tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Test Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "F1-score:", 2*p*r/(p+r)



print "Start close testing...", datetime.datetime.now()
results = []
for line in closetestdata:
    x, yref = util.line_toseq(line, charstop)
    out = hmmtagger.tag(x)
    _, yout = zip(*out)
    results.append(util.eval(yref, yout, "S"))
tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Training Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "F1-score:", 2*p*r/(p+r)
