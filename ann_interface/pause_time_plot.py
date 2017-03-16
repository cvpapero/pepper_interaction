# coding:utf-8

import argparse

import numpy as np
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
        hist[round(hmin/res)+round(h/res)-1]+=1
    return hist


def barh_plot(data1, data2, time, n=0):
    #data1(1000,1)
    start_1, start_2 = 0, 0
    width = 0.9
    data1_y, data2_y = 1, 0


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
    #print len(times), len(data1)#times

    
    plt.xlim(0, len(data1))
    plt.ylim(-1, 2)
    if n < 3:
        plt.yticks([data2_y, data1_y],["robot","person"])
    else:
        plt.yticks([data2_y, data1_y],["person'","person"])
        
    plt.xticks(np.arange(len(data1))[::500], np.round(times[::500]))
    plt.xlabel("time[sec]")
    plt.grid()
    plt.rcParams["font.size"] = 10
    #plt.show()


def calc_interval(data, diff_time):
    
    start = 0
    sum_time = 0
    for t in range(len(data)-1):
        if data[t] == 0 and data[t+1] == 1:
            start = t
        if data[t] == 1 and data[t+1] == 0:
            sum_time += sum(diff_time[start:t])
            
    return sum_time
    
fnames = sorted(glob.glob(args.filename+"/*"))

cap = ["person", "robot"]
types = ["model", "control", "random", "person"]



for n, fname in enumerate(fnames):

    input_data = data_proc2.load_proced_data_flag([fname], datalen=-1) #(1000, 2)
    a_data = np.abs(input_data["annos"])#[:1000]
    t_data = input_data["times"]#[:1000]

    diff_time = []
    for i in range(len(t_data)-1):
        if i == 0:
            diff_time.append(0)
        else:
            diff_time.append(t_data[i+1]-t_data[i])


    plt.subplot(4,1,n+1)

    if n < 3:
        barh_plot(a_data[:,0], a_data[:,1], t_data)
        #発話合計時間計算
        sum_time_a1 = calc_interval(a_data[:,0], diff_time)
        sum_time_a2 = calc_interval(a_data[:,1], diff_time)
    else:
        barh_plot(a_data[:,1], a_data[:,0], t_data, n=3)
        sum_time_a1 = calc_interval(a_data[:,1], diff_time)
        sum_time_a2 = calc_interval(a_data[:,0], diff_time)
    
    plt.title(types[n]+"_speaker interval", fontsize=12)

    print sum_time_a1, sum_time_a2


    """
    #print diff_time
    p1_times, p1_silences, p1_ignores = pause_time_check(a_data[:,1], a_data[:,0], diff_time)
    p2_times, p2_silences, p2_ignores = pause_time_check(a_data[:,0], a_data[:,1], diff_time)


    print len(p1_times), len(p2_times)
    print "person",np.round(p1_times,4)
    print "robot",np.round(p2_times,4)

    print len(p1_silences), len(p2_silences)
    print np.round(p1_silences,4)
    print np.round(p2_silences,4)
    
    print len(p1_ignores), len(p2_ignores)
    print np.round(p1_ignores,4)
    print np.round(p2_ignores,4)

    
    hmax, hmin = 30, 5
    ymax = 8
    res = 0.5
    pick = 10
    hist1 = histogram(p1_times, hmax, hmin, res)
    hist2 = histogram(p2_times, hmax, hmin, res)
     
    datarange = round(hmax/res)+round(hmin/res)+1
    label = np.hstack((-res*np.arange(1,int(hmin/res)+1)[::-1],0,res*np.arange(1, int(hmax/res)+1)))#[::5]
    locate = np.arange(datarange)

    plt.subplot(len(fnames), 2, n*2+1)
    plt.bar(locate, hist1, tick_label=locate, align='center')
    plt.xticks(locate[::pick],label[::pick])
    plt.xlim(0,datarange)
    plt.ylim(0,ymax)
    plt.grid()
    title = cap[0]+":"+types[n]+"_count:"+str(len(p1_times))+"_ignored:"+str(len(p1_ignores))+"_continue:"+str(len(p1_silences))
    plt.title(title, fontsize=12)
    
    plt.subplot(len(fnames), 2, n*2+2)
    plt.bar(locate, hist2, tick_label=locate, align='center')
    plt.xticks(locate[::pick],label[::pick])
    plt.xlim(0,datarange)
    plt.ylim(0,ymax)
    plt.grid()
    cap = "person'" if types[n] == "person" else "robot"#cap[1]
    title = cap+":"+types[n]+"_count:"+str(len(p2_times))+"_ignored:"+str(len(p2_ignores))+"_continue:"+str(len(p2_silences))
    plt.title(title, fontsize=12)

    plt.rcParams["font.size"] = 8
    """
plt.tight_layout()
plt.show()




