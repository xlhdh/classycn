# -*- coding: utf8 -*-
import math
puncts = "\t .!?#$%&\'()*+,-/:;<=>@[\]^_`{|}~！？｡。'\"＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏'·".decode('utf8')

def file_to_lines(filenames):
    for fn in filenames:
        file = open(fn, 'r')
        for line in file:
            line = line.decode('utf8').replace('\n',"")
            if len(line)>0:
                yield line
        file.close()

def line_toseq(line, charstop):
    if charstop: return char_stop_toseq(line)
    else: return stop_char_toseq(line)

def make_charset(list_of_sentences):
    li = [line_toraw(line) for line in list_of_sentences]
    return list(set(list("".join(li))))

def line_toraw(line):
    import re
    return re.sub('[%s]' % re.escape(puncts+"\n".decode('utf8')), '', line)

def seq_to_line(x, y, charstop):
    assert len(x)==len(y)
    s = ""
    for a, b in zip(x, y):
        print a
        print b
        if b =='S': b=','
        else: b=''
        if charstop: s = s+a+b
        else: s = s+b+a
    return s

def decode_totext(rawtext, outputs, charstop):
    assert len(rawtext) == len(outputs)
    text = []
    for i in range(len(rawtext)):
        x = rawtext[i]
        y = outputs[i]
        if charstop: y = y[1:]
        else: y = y[:-1]
        line = ""
        #print x, y, len(x), len(y)
        assert len(x) == len(y)
        for j in range(len(x)):
            if y[j][0]:
                if charstop: line = line + x[j] + "."
                else: line = line + "."+x[j]
            else: line = line + x[j]
        #line = line + x[j]
        text.append(line)
    return text

def seq_to_sparsevec(x,y,charset):
    #input: x, y: [char,...], [label,...]
    xseq = []
    for c in x:
        vec = [0] * len(charset)
        if c in charset: vec[charset.index(c)] = 1
        xseq.append(vec)

    yseq = []
    for l in y:
        if l == 'N': vec = [0]
        else: vec = [1]
        yseq.append(vec)

    return xseq, yseq

def seq_to_densevec(x, y, mydict):
    #xseq = [mydict.get(c, default=mydict["zero"]) for c in x]
    xseq = []
    for c in x:
        if mydict.has_key(c):
            xseq.append(mydict[c])
        else:
            xseq.append(mydict["zero"])

    yseq = []
    for l in y:
        if l == 'N': vec = [0]
        else: vec = [1]
        yseq.append(vec)
    
    return xseq, yseq


def char_stop_toseq(line):
    # attributes segmentation to the preceding character
    line = "S,"+line
    c = []
    l = []
    for u in line:
        if u in puncts:
            l[-1]="S"
        else:
            c.append(u)
            l.append("N")
    return c,l

def stop_char_toseq(line):
    # attributes segmentation to the following character
    line = line+",E"
    c = []
    l = []
    currentlabel = "N"
    for u in line:
        if u in puncts:
            currentlabel = "S"
        else:
            c.append(u)
            l.append(currentlabel)
            currentlabel = "N"
    assert len(c) == len(l)
    return c,l

def readvec(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        gram = line[0].decode('utf8')
        params = line[1:]
        params = [float(p) for p in params]
        base = math.sqrt(sum([p*p for p in params]))
        #params = [p/base for p in params]
        
        data  = {}
        for i in range(len(params)):
            data[str(i)] = params[i]/base
        
        mydict[gram] = data
    #mydict[gram] = params
    file.close()
    return mydict

def readvec2(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        gram = line[0].decode('utf8')
        params = line[1:]
        params = [float(p) for p in params]
        data  = {}
        for i in range(len(params)):
            data[str(i)] = params[i]
        
        mydict[gram] = data
    file.close()
    return mydict

def lstmvec(vecfilename):
    mydict = {}
    file = open(vecfilename, 'r')
    for line in file:
        line = line.split()
        gram = line[0].decode('utf8')
        params = line[1:]
        params = [float(p) for p in params]
        base = math.sqrt(sum([p*p for p in params]))
        params = [p/base for p in params]
        
        mydict[gram] = params
    zr = [p*0 for p in params]
    mydict["zero"] = zr
    file.close()
    return mydict

def eval(ref, out, signi):
    assert len(ref) == len(out)
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    for i in range(len(ref)):
        if ref[i] == signi:
            if out[i] == signi: tp = tp+1
            else: fn = fn+1
        elif out[i] == signi:
            fp = fp+1
        else:
            tn = tn+1
    return tp, fp, fn, tn


