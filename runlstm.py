# -*- coding: UTF8 -*-
import lstm
import datetime
import util
import glob
import numpy
import random
import sys

material = 'data/sjw/*'
size = 80
trainportion = 0.8
validateportion = 0.1
cut1 = int(size*trainportion)
cut2 = int(size*(trainportion+validateportion))
dictfile = 'data/vector/sjwcbow50.txt'
dense = True # 1 = dense, 0 = one-hot sparse
charstop = True # True means label attributes to previous char
modelname = material.replace('/','').replace('*','')+str(size)
validate_interval = 1000
hidden_size = 50
learning_rate = 0.001
random.seed(101)

print "Material:", material
print "Size:", size, "entries,", trainportion, "as training", validateportion, "as validation"

starttime = datetime.datetime.now()
print "Starting Time:",starttime

print "Preparing text..."
li = [line for line in util.file_to_lines(glob.glob(material))]
random.shuffle(li)
li = li[:size]

print "Preparing dictionaries..."
if dense: vdict = util.lstmvec(dictfile)
else: charset = util.make_charset(li)

print "Preparing datasets..."
dataset = []
for line in li:
    x, y = util.line_toseq(line, charstop)
    if dense: dataset.append(util.seq_to_densevec(x, y, vdict))
    else: dataset.append(util.seq_to_sparsevec(x,y,charset))
dataset_train = dataset[:cut1]
dataset_validate = dataset[cut1:cut2]
dataset_test = dataset[cut2:]
li_generate = [util.line_toraw(line) for line in li[cut2:]]

#sys.exit()

min_val_loss = 1000000 # very big
peak = 0
int_num = 0

mylstm = lstm.LSTM(n_input=len(dataset_test[0][0][0]),n_output=len(dataset_test[0][1][0]),n_memblock=hidden_size, lr=learning_rate)
#mylstm.load("m50saving1740")

try:
    while True:
        numpy.random.shuffle(dataset_train)
        dt = [dataset_train[x:x+validate_interval] for x in xrange(1, len(dataset_train), validate_interval)]
        for d in dt:
            mylstm.train(d)
            vcost, act, aco, atp, p, r, f = mylstm.test(dataset_validate)
            mylstm.save(modelname + "saving" + str(int_num))
            if vcost < min_val_loss:
                min_val_loss = vcost
                peak = int_num
            int_num = int_num + 1
            print int_num, "@@VALIDATE ON VALIDATE@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", vcost
            print int_num, "@@VALIDATE ON VALIDATE@@\t", "P, R, F:", p, r, f
            print int_num, "@@VALIDATE ON VALIDATE@@\t", datetime.datetime.now(), datetime.datetime.now()-starttime
        #print "Sample value: ", mylstm.generate([dt[0],])
        tcost, act, aco, atp, p, r, f = mylstm.test(dataset_train)
        print "\t@@VALIDATE ON TRAIN@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", tcost
        print "\t@@VALIDATE ON TRAIN@@\tP, R, F:", p, r, f
        print "\t@@VALIDATE ON TRAIN@@\t", datetime.datetime.now(), datetime.datetime.now()-starttime
except KeyboardInterrupt:
    print 'Interrupted by user.'



#This is the final test.
print "This is the test for the PEAK value."
mylstm.load(modelname + "saving" + str(peak))
testcost, act, aco, atp, p, r, f = mylstm.test(dataset_test)
print "\t@@VALIDATE ON TEST@@\tTotal in Gold:", act, "Total in Output:", aco, "True Positive:", atp, "Loss:", testcost
print "\t@@VALIDATE ON TEST@@\tP, R, F:", p, r, f
print "\t@@VALIDATE ON TEST@@\tTotal in Gold:", datetime.datetime.now(), datetime.datetime.now()-starttime
print "\tpeak =", peak


result = util.decode_totext(li_generate, mylstm.generate(dataset_test), charstop)
for line in result:
    print line.encode('utf8')



