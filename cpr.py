# -*- coding: utf8 -*-
import sys
import glob
import nltk.tag as nt
import util

" PARAMETERS "
charstop = False # True means label attributes to previous char
" END OF PARAMETERS "

"python cpr.py 'qualitative/allover-sjw-gold.txt' 'qualitative/allover-sjw-me.txt' 1"
args = sys.argv
if len(args)>1:
    material1 = args[1]
    material2 = args[2]
    charstop = int(args[3])

# Prepare li: list of random lines
print "Reading from files..."
gold = [line for line in util.file_to_lines(glob.glob(material1))]
out = [line for line in util.file_to_lines(glob.glob(material2))]


golddata = []
for line in gold:
    golddata.append(util.line_toseq(line, charstop))

outdata = []
for line in out:
    outdata.append(util.line_toseq(line, charstop))

# testdata shape: [([x1, x2, ...],[y1,y2,...]),([],[])]

results = []
assert len(golddata)==len(outdata)
for i in range(len(golddata)):
    try:
        yref = golddata[i][1]
        yout = outdata[i][1]
        results.append(util.eval(yref, yout, "S"))
    except AssertionError:
        print i+1
        print len(yref)#, yref
        print len(yout)#, yout
        xref = golddata[i][0]
        xout = outdata[i][0]
        for i in range(len(xref)):
            if xref[i] != xout[i]:
                print xref[i]
                pass

tp, fp, fn, tn = zip(*results)
tp, fp, fn, tn = sum(tp), sum(fp), sum(fn), sum(tn)

p, r = tp/(tp+fp), tp/(tp+fn)
print "Total tokens in Test Set:", tp+fp+fn+tn
print "Total S in REF:", tp+fn
print "Total S in OUT:", tp+fp
print "Presicion:", p
print "Recall:", r
print "F1-score:", 2*p*r/(p+r)
