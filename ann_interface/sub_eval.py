#!/usr/bin/python
# -*- coding: utf-8 -*-



import argparse

import numpy as np
import json
import matplotlib.pyplot as plt
import data_proc2


parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', default="yamakoshi_0125/0125_mimic_anno.json", 
                    help='filename')
args = parser.parse_args()

label = ["control", "random", "model"]
titles = ["person A", "person B", "person C", "person D"]



yamakoshi=[[1], [17,1,7,18,17,2,4,15,4,5], [3,8,2,5]]
hirai=[[2], [1,1,5,3,2,2,2,2,3,2], [1]]
yamamoto=[[1,1,1,2,1,1,1], [14, 23,7, 9, 10, 4, 3, 17,1, 6, 1], [1, 2,6,1,1,2]]
yano=[[5, 2, 1, 2],[17,1,7,5,6],[2,1,7,3,12,3,6,1]]

persons = [yamakoshi, hirai, yamamoto, yano]

sbs=[]
for i, data in enumerate(persons):

    sb = []
    for d in data:
        sb.append(sum(d))
    sbs.append(sb)

    
sbs = np.array(sbs)
print sbs
sums = np.average(sbs, axis=0)
print sums
plt.bar(np.arange(len(sums)), sums, tick_label=label, align='center')
plt.ylabel("average of times[sec]")
plt.ylim(0,70)
plt.title("evaluation")
plt.grid()
plt.show()

"""
yamakoshi=[[1], [17,1,7,18,17,2,4,15,4,5], [3,8,2,5]]
hirai=[[2], [1,1,5,3,2,2,2,2,3,2], [1]]
yamamoto=[[1,1,1,2,1,1,1], [14, 23,7, 9, 10, 4, 3, 17,1, 6, 1], [1, 2,6,1,1,2]]

persons = [yamakoshi, hirai, yamamoto]
"""
for i, data in enumerate(persons):
    plt.subplot(1,len(persons),i+1)

    datanum = len(data)
    viz_data=[sum(data[0]),sum(data[1]),sum(data[2])]
    plt.bar(np.arange(datanum), viz_data, tick_label=label, align='center')
    plt.ylabel("sum of times[sec]")
    plt.ylim(0,100)
    plt.title(titles[i]+" evaluation", fontsize=11)
    plt.grid()
plt.tight_layout()
plt.show()

