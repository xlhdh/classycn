# -*- coding: utf8 -*-
import util
import sys
import glob

material = '../data/24s/*'
#material = 'data/24s/*'
i=0
for line in util.file_to_lines(glob.glob(material)):
    #i = i+1
    print " ".join(util.line_toraw(line)).encode('utf8')
#print i



