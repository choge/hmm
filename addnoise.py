#!/usr/bin/env python
# -*- coding:utf-8 -*-

import ghmm
import hmm
import predictor
import pickle

# make a dataset
dm = predictor.FastaDataSetMaker()
ta = dm.read_from_file("TA_all.fasta",
        name="ta", label=1)
charlist = "ACDEFGHIKLMNPQRSTVWY"
char_dic = {charlist[i]: i for i in xrange(20)}
ta_n = [[char_dic[c] for c in seq.sequence[::-1]] for seq in ta]

# convert hmm
g = ghmm.HMMOpen("ta2.1.2.xml")
h = hmm.convert_ghmm(g)

# Training
h.baum_welch(ta_n, iter_limit=1000, threshold=-1e-3,
        pseudocounts=[0, 1e-3, 0])
g2 = hmm.convert2ghmm(h)
f = open("ta_with_noise", "w")
pickle.dump(g2, f)

