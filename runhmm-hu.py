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

"python runhmm-hu.py 'data/sjw/*' 1000000 1 qualitative/allover-sjw-gold.txt"
"python runhmm-hu.py 'data/24s/*' 10000 1 qualitative/allover-24s-gold.txt"
args = sys.argv
if len(args)>1:
    material = args[1]
    size = int(args[2])
    charstop = int(args[3])
    hu = args[4]
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
lines = []
testdata = [line for line in util.file_to_lines(glob.glob(hu))]
for line in testdata:
    x, yref = util.line_toseq(line, charstop)
    out = hmmtagger.tag(x)
    _, yout = zip(*out)
    results.append(util.eval(yref, yout, "S"))
    lines.append(util.seq_to_line(x,yout,charstop))
tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Test Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "F1-score:", 2*p*r/(p+r)

while lines:
    print lines.pop().encode('utf8')
