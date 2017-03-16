# coding:utf-8

import argparse

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import data_proc2
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--filename', '-f', default='yamakoshi_manual', help='filename')
args = parser.parse_args()


def pause_time_check(a1_data, a2_data, diff_time):
    times = []
    silences = []
    ignores = []
    datalen=len(a1_data)
    start = 0
    for t in range(datalen-1):

        if a1_data[t] == 0 and a1_data[t+1] == 1:
            start = t
            
        
        if a1_data[t] == 1 and a1_data[t+1] == 0:
            #p1発話終ったとき
            time = 0
            if a2_data[t] == 0:
                #p2発話まで間があるなら
                #print "s"
                for s in range(t, datalen-1):
                    time += diff_time[s]
                    if a2_data[s] == 1:
                        times.append(time)
                        break
                    if a1_data[s+1] == 1:
                        silences.append(time)
                        break
                    #if s == datalen-2:
            else:
                #おーばーらっぷしてるなら
                #print "o"
                for s in reversed(range(t)):
                    time -= diff_time[s]
                    if a2_data[s] == 0:
                        times.append(time)
                        break
                    if s < start:
                        ignores.append(time)
                        break
                    
    return times, silences, ignores


def histogram(data, hmax, hmin, res):
    hist = np.zeros(round(hmax/res)+round(hmin/res)+1)
    for h in data:

        #num = 
        num = round(hmin/res,2)+round(h/res,2)
        hist[num]+=1
        #print round(hmin/res,1)+round(h/res,1)
    return hist


def barh_plot(data1, data2, time, n=0, title=""):
    #data1(1000,1)
    start_1, start_2 = 0, 0
    width = 0.9
    data1_y, data2_y = 1, 0

    plt.subplot(4,1,n+1)
    for i in range(len(data1)-1):
        if data1[i]==0 and data1[i+1]==1:
            start_1 = i
        
        if data2[i]==0 and data2[i+1]==1:
            start_2 = i
            
        if data1[i]==1 and data1[i+1]==0:
            plt.barh(data1_y, i-start_1, width, left=start_1, color='r', edgecolor='k', align="center")

        if data2[i]==1 and data2[i+1]==0:
            plt.barh(data2_y, i-start_2, width, left=start_2, color='g', edgecolor='k', align="center")

    times = []
    for i in range(len(time)):
        times.append(time[i]-time[0])
    print len(times), len(data1)#times
    plt.xlim(0, len(data1))
    plt.ylim(-1, 2)
    plt.yticks([data2_y, data1_y],["robot","person"])
    plt.xticks(np.arange(len(data1))[::500], np.round(times[::500]))
    plt.xlabel("time[sec]")
    plt.title(title+"_speaker/listener result", fontsize=12)

    plt.rcParams["font.size"] = 10
    #plt.show()

def calc_diff_time(time):
    diff_time = []
    for i in range(len(t_data)-1):
        if i == 0:
            diff_time.append(0)
        else:
            diff_time.append(t_data[i+1]-t_data[i])
    return diff_time


hmax, hmin = 30, 5
ymax = 16
res = 0.5
pick = 10
types = ["model", "control", "random", "person"]

fnames_list = ["yamakoshi_manual", "hirai_manual", "yamamoto_manual","yano_manual"]
#fnames_list = ["yamakoshi_manual", "yano_manual"]
#fnames_list = ["yamakoshi_manual", "yamamoto_manual","yano_manual"]


user_hists1, user_hists2 = [], []
#user_igd1, user_igd2 = [], []
#user_cont1, user_cont2 = [], []

p1_times_stack=[[],[],[],[]]
p2_times_stack=[[],[],[],[]]

igd1, igd2 = [0, 0, 0, 0], [0, 0, 0, 0]
cont1, cont2 = [0, 0, 0, 0], [0, 0, 0, 0]

for fnames_l in fnames_list:
    print fnames_l 
    fnames = sorted(glob.glob(fnames_l+"/*"))
    hists1, hists2 = [], []
    p1_stack, p2_stack=[],[]
    for n, fname in enumerate(fnames):
        input_data = data_proc2.load_proced_data_flag([fname], datalen=-1) #(1000, 2)
        a_data = np.abs(input_data["annos"])#[:1000]
        t_data = input_data["times"]#[:1000]

        
        diff_time = calc_diff_time(t_data)

        p1,p2 = 1, 0
        if n>2:
            p1,p2 = 0, 1
        #print p1, p2
        p1_times, p1_silences, p1_ignores = pause_time_check(a_data[:,p1], a_data[:,p2], diff_time)
        p2_times, p2_silences, p2_ignores = pause_time_check(a_data[:,p2], a_data[:,p1], diff_time)

        #print "p1_times",p1_times
        p1_stack.append(p1_times)
        p2_stack.append(p2_times)
        p1_times_stack[n].extend(p1_times)
        p2_times_stack[n].extend(p2_times)
        #print len(p1_times)
        # print "ignored person", np.round(p2_ignores,4), len(p2_ignores)
        #print "ignored robot", np.round(p1_ignores,4), len(p1_ignores)
 
        #print "continue person", np.round(p2_silences,4), len(p2_silences)
        #print "continue robot",np.round(p1_silences,4), len(p1_silences)
        


        hist1 = histogram(p1_times, hmax, hmin, res)
        hist2 = histogram(p2_times, hmax, hmin, res)
        #print hist1.shape
        #print hist2.shape
        hists1.append(hist1)
        hists2.append(hist2)
        
        cont1[n] += len(p2_silences)
        cont2[n] += len(p1_silences)
        igd1[n] += len(p2_ignores)
        igd2[n] += len(p1_ignores)
        
    user_hists1.append(hists1)
    user_hists2.append(hists2)

    #個別に検定
    p1_model = p1_stack[2]
    for n, p1 in enumerate(p1_stack):
        if n==2:
            continue
        result = stats.mannwhitneyu(p1, p1_model)
        print "p1 model",types[n],result.pvalue

    
    p2_model = p2_stack[2]
    for n, p2 in enumerate(p2_times_stack):
        if n==2:
            continue
        result = stats.mannwhitneyu(p2, p2_model)
        print "p2 model",types[n],result.pvalue
    

    
user1 = np.array(user_hists1)
user2 = np.array(user_hists2)

sum1 = np.sum(user1, axis=0)
sum2 = np.sum(user2, axis=0)

"""
#U検定
p1_model = p1_times_stack[2]
p2_model = p2_times_stack[2]
result = stats.mannwhitneyu(p1_model, p2_model)
print result.pvalue
"""

p1_aves, p2_aves = [], []
p1_meds, p2_meds = [], []
p1_errs, p2_errs = [], []


for n,(p1,p2) in enumerate(zip(p1_times_stack,p2_times_stack)):
    #stats.probplot(p1, dist="norm", plot=pylab)
    #result1 = stats.shapiro(p1)
    #result2 = stats.shapiro(p2)
    ave1, ave2 = np.average(p1), np.average(p2)
    med1, med2 = np.median(p1), np.median(p2)
    
    #sd1, sd2 = np.std(p1)/np.sqrt(len(p1)), np.std(p2)/np.sqrt(len(p2))
    sd1, sd2 = np.std(p1), np.std(p2)

    #spc1, spc2 = stats.scoreatpercentile(p1, 25), stats.scoreatpercentile(p2, 25)
    #print types[n],":ave1, std1, med1:", ave1, sd1, med1, " ave2, std2, med2:", ave2, sd2, med2
    print types[n],"med:", med1, med2 
    p1_aves.append(ave1)
    p2_aves.append(ave2)
    p1_errs.append(sd1)
    p2_errs.append(sd2)

    p1_meds.append(med1)
    p2_meds.append(med2)


ymine, ymaxe = -4, 12

plt.subplot(1,2,1)
plt.plot(p1_aves, 'ro')
plt.xticks([0,1,2,3],types)
plt.errorbar(np.arange(len(p1_aves)),p1_aves, fmt='ro',ecolor='g', yerr=p1_errs)
plt.yticks(np.arange(ymine, ymaxe), np.arange(ymine, ymaxe))
plt.xlim(-0.5, len(p1_aves)-0.5)
plt.ylim(ymine, ymaxe)
plt.ylabel("time[sec]")
plt.grid()


plt.subplot(1,2,2)
plt.plot(p2_aves, 'ro')
#plt.plot(p2_meds, 'bo')
plt.xticks([0,1,2,3],types)
plt.errorbar(np.arange(len(p2_aves)),p2_aves, fmt='ro',ecolor='g', yerr=p2_errs)
plt.yticks(np.arange(ymine, ymaxe), np.arange(ymine, ymaxe))
plt.xlim(-0.5, len(p2_aves)-0.5)
plt.ylim(ymine, ymaxe)
plt.ylabel("time[sec]")
plt.grid()


plt.show()
    
"""
p1_model = p1_times_stack[2]
for n,p1 in enumerate(p1_times_stack):
    if n==2:
        continue
    #w, p = stats.wilcoxon(p1, p1_model)
    #print "p1 model",types[n],p
    result = stats.mannwhitneyu(p1, p1_model)
    print "p1 model",types[n],result.pvalue
    #t, p = stats.ttest_rel(p1, p1_model)
    #print "ttest p1 model",types[n], p
    
p2_model = p2_times_stack[2]
for n,p2 in enumerate(p2_times_stack):
    if n==2:
        continue
    result = stats.mannwhitneyu(p2, p2_model)
    print "p2 model",types[n],result.pvalue


result = stats.mannwhitneyu(p1_times_stack[0], p1_times_stack[3])
print "p1 control person",result.pvalue


result = stats.mannwhitneyu(p2_times_stack[0], p2_times_stack[3])
print "p2 control person",result.pvalue


pstacks = []
for p1, p2 in zip(p1_times_stack, p2_times_stack):
    #print p1+p2
    pstacks.append(p1+p2)
"""
#print pstacks
"""   
p_model = pstacks[2]
for n, p in enumerate(pstacks):
    #if n==2:
    #    continue
    result = stats.mannwhitneyu(p, p_model)
    print "sum: model",types[n], result.pvalue
"""

#print user1.shape
#print sum1.shape
#print np.sum(sum1, axis=1)

#print user_igd1
#print user_igd2
#print user_cont1
#print user_cont2

#user_igd1 = np.array(user_igd1)
#user_igd2 = np.array(user_igd2)
#user_cont1 = np.array(user_cont1)
#user_cont2 = np.array(user_cont2)

#print igd1
#print igd2
#print cont1
#print cont2

"""
n=0
for n, (h1, h2) in enumerate(zip(sum1, sum2)):
    print n
    datarange = round(hmax/res)+round(hmin/res)+1
    label = np.hstack((-res*np.arange(1,int(hmin/res)+1)[::-1],0,res*np.arange(1, int(hmax/res)+1)))#[::5]
    locate = np.arange(datarange)

    plt.subplot(len(fnames), 2, n*2+1)
    plt.bar(locate, h1, tick_label=locate)
    plt.xticks(locate[::pick],label[::pick])
    plt.xlim(0,datarange)
    plt.ylim(0,ymax)
    plt.xlabel("time[sec]")
    plt.ylabel("num")
    plt.grid()
    title = "person:"+types[n]+"_ignored:"+str(igd1[n])+"_continue:"+str(cont1[n])
    plt.title(title, fontsize=12)
    
    plt.subplot(len(fnames), 2, n*2+2)
    plt.bar(locate, h2, tick_label=locate)
    plt.xticks(locate[::pick],label[::pick])
    plt.xlim(0,datarange)
    plt.ylim(0,ymax)
    plt.xlabel("time[sec]")
    plt.ylabel("num")
    plt.grid()
    cap = "person'" if types[n] == "person" else "robot"#cap[1]
    title = cap+":"+types[n]+"_ignored:"+str(igd2[n])+"_continue:"+str(cont2[n])
    plt.title(title, fontsize=12)
    plt.rcParams["font.size"] = 8
    
plt.tight_layout()
plt.show()
"""




    

