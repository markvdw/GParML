#!/usr/bin/python
'''
pre_process.py

Pre-process data given in a CSV format to match expected input from parallel_GPLVM

We get as an input a CSV file name ''file_name'' and number of parts ''P'', and distribute the data uniformly
to P different input files named ''file_name_i''. The input file is NOT read into memory but rather streamed to
support large files.

In future versions, this file will support centering and scaling of the data as well.
'''
import sys
from random import choice

if not len(sys.argv) == 3:
	raise Exception('Input format: file_name P')

file_name = sys.argv[1]
P = sys.argv[2]

f=[]
for i in xrange(1,int(P) + 1):
	name = file_name + '_' + str(i)
	f.append(open(name,'w'))

with open(file_name, 'r') as inpit_file:
	for line in inpit_file:
		choice(f).write(line)

