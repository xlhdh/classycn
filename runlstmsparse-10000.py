# -*- coding: UTF8 -*-
import lstm
import datetime
import util
import glob
import numpy
import random
import sys

material = 'data/sjw/*'
size = 10000
trainportion = 0.9
validateportion = 0.05
cut1 = int(size*trainportion)
cut2 = int(size*(trainportion+validateportion))
dictfile = 'data/vector/sjwcbow50.txt'
dense = False# 1 = dense, 0 = one-hot sparse
charstop = True # True means label attributes to previous char
modelname = material.replace('/','').replace('*','')+str(size)+"sparse"
validate_interval = 1000
hidden_size = 50
learning_rate = 0.001
random.seed(101)

print "Material:", material
print "Size:", size, "entries,", trainportion, "as training", validateportion, "as validation"
print "Dense:", dense
print "charstop:", charstop

starttime = datetime.datetime.now()
print "Starting Time:",starttime

print "Preparing text..."
li = [line for line in util.file_to_lines(glob.glob(material))]
random.shuffle(li)
li = li[:size]

print "Preparing dictionaries..."
if dense: vdict = util.lstmvec(dictfile)
else: charset = util.make_charset(li,3)

print "Preparing datasets..."

dataset_train = li[:cut1]
dataset_validate = li[cut1:cut2]
dataset_test = li[cut2:]

dataset = []
while dataset_train:
    x, y = util.line_toseq(dataset_train.pop(), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_train)%1000: print "len(dataset_train)", len(dataset_train)
dataset_train = dataset

dataset = []
while dataset_validate:
    x, y = util.line_toseq(dataset_validate.pop(), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_validate)%1000: print "len(dataset_validate)", len(dataset_validate)
dataset_validate = dataset


#sys.exit()

min_val_loss = float("inf") # very big
peak = 0
int_num = 0

print "Making LSTM..."
#print dataset_train[0][0].shape
#print dataset_train[0][0][0].shape
if dense: i,o = len(dataset_train[0][0][0]),len(dataset_train[0][1][0])
else: i,o = dataset_train[0][0].shape[1],dataset_train[0][1].shape[1]
mylstm = lstm.LSTM(n_input=i, n_output=o, n_memblock=hidden_size, lr=learning_rate, dense=dense)
#mylstm.load("m50saving1740")

print "Start Training... "
try:
    while True:
        numpy.random.shuffle(dataset_train)
        dt = [dataset_train[x:x+validate_interval] for x in xrange(1, len(dataset_train), validate_interval)]
        for d in dt:
            mylstm.train(d)
            vcost, act, aco, atp, p, r, f = mylstm.test(dataset_validate)
            mylstm.save(modelname + "/saving-" + str(int_num))
            if vcost < min_val_loss:
                min_val_loss = vcost
                peak = int_num
            int_num = int_num + 1
            print int_num, "@@VALIDATE ON VALIDATE@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", vcost
            print int_num, "@@VALIDATE ON VALIDATE@@\t", "P, R, F:", p, r, f
            print int_num, "@@VALIDATE ON VALIDATE@@\t", datetime.datetime.now(), datetime.datetime.now()-starttime
        #print "Sample value: ", mylstm.generate([dt[0],])
        tcost, act, aco, atp, p, r, f = mylstm.test(dataset_train[:100])
        print "\t@@VALIDATE ON TRAIN-100@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", tcost
        print "\t@@VALIDATE ON TRAIN-100@@\tP, R, F:", p, r, f
        print "\t@@VALIDATE ON TRAIN-100@@\t", datetime.datetime.now(), datetime.datetime.now()-starttime
except KeyboardInterrupt:
    print 'Interrupted by user.'


dataset = []
while dataset_test:
    x, y = util.line_toseq(dataset_test.pop(0), charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_test)%1000: print "len(dataset_test)", len(dataset_test)
dataset_test = dataset

#This is the final test.
print "This is the test for the PEAK value."
mylstm.load(modelname + "/saving-" + str(peak))
testcost, act, aco, atp, p, r, f = mylstm.test(dataset_test)
print "\t@@VALIDATE ON TEST@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", testcost
print "\t@@VALIDATE ON TEST@@\tP, R, F:", p, r, f
print "\t@@VALIDATE ON TEST@@\tTotal in Gold:", datetime.datetime.now(), datetime.datetime.now()-starttime
print "\tpeak =", peak

generate = [line for line in util.file_to_lines(glob.glob('qualitative/allover-sjw-gold*'))]
dataset_generate = []
for line in generate:
    x, y = util.line_toseq(line, charstop)
    if dense: dataset_generate.append(util.seq_to_densevec(x, y, vdict))
    else: dataset_generate.append(util.seq_to_sparsevec(x,y,charset))
    if not len(dataset_generate)%1000: print "len(dataset_generate)", len(dataset_generate)

generate = [util.line_toraw(line) for line in generate]

result = util.decode_totext(generate, mylstm.generate(dataset_generate), charstop)
for line in result:
    print line.encode('utf8')

