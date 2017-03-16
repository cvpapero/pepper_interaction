#!/usr/bin/python
#coding: utf-8

import math
import audioop
import rospy
from std_msgs.msg import Float64
from humans_msgs.msg import AudioData

#プロット関係のライブラリ
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import sys

#音声関係のライブラリ
import pyaudio
import struct


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-d', default="",
                    help='device name')
parser.add_argument('--topic', '-t', default=0, type=int,
                    help='topic number [0/1]')
args = parser.parse_args()



class PlotWindow:
    def __init__(self, device_id, device_rate):

        rospy.init_node("audio_"+str(device_id)+"-"+str(device_rate))

        #きめうち
        #topic_id = 1 if device_id == 5 else 2
        #topic_id = 1 if device_id == 0 else 2
        topic_id = args.topic
        
        self.apub = rospy.Publisher('/humans/audio/'+str(topic_id), AudioData, queue_size=10)

        
        #プロット初期設定
        self.win=pg.GraphicsWindow()
        self.win.setWindowTitle(u"リアルタイムプロット")
        self.plt=self.win.addPlot() #プロットのビジュアル関係
        self.plt.setYRange(-1,1)    #y軸の上限、下限の設定
        self.curve=self.plt.plot()  #プロットデータを入れる場所

        #マイクインプット設定
        self.CHUNK=1024             #1度に読み取る音声のデータ幅
        self.RATE=device_rate             #サンプリング周波数
        self.audio=pyaudio.PyAudio()
        self.stream=self.audio.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=self.RATE,
                                    input=True,
                                    input_device_index=device_id, 
                                    frames_per_buffer=self.CHUNK)

        #アップデート時間設定
        self.timer=QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)    #10msごとにupdateを呼び出し

        #音声データの格納場所(プロットデータ)
        self.data=np.zeros(self.CHUNK)

    def update(self):
        #print self.data.shape
        wave, volume = self.AudioInput()
        self.data=np.append(self.data, wave)
        if len(self.data)/1024 > 1:     #5*1024点を超えたら1024点を吐き出し
            self.data=self.data[1024:]
        self.curve.setData(self.data)   #プロットデータを格納

 
        score = volume #sum(np.fabs(self.data))/len(self.data)
        #score = max(np.fabs(self.data))
        #scorelen=100
        #score = sum(sorted(abs(self.data)).reverse()[0:scorelen])/scorelen
        
        print "max:",score
        amsg = AudioData()       
        amsg.header.stamp = rospy.get_rostime()
        amsg.data = score#1 if ave > 0.005 else 0 
        self.apub.publish(amsg)

        
    def AudioInput(self):
        ret=self.stream.read(self.CHUNK)    #音声の読み取り(バイナリ)
        rms = audioop.rms(ret, 2)
        #decibel = 20*math.log10(rms)
        #バイナリ → 数値(int16)に変換
        #32768.0=2^16で割ってるのは正規化(絶対値を1以下にすること)
        ret=np.frombuffer(ret, dtype="int16")/32768.0
        return ret, rms#decibel


def input_device():
    p_in = pyaudio.PyAudio()
    print "device num: {0}".format(p_in.get_device_count())
    for i in range(p_in.get_device_count()):
        print i,":",p_in.get_device_info_by_index(i)["name"]
    print "input use device:"
    d_id = int(raw_input())
    #print "input topic id(0/1):"
    #t_id = int(raw_input())
    return d_id, int(p_in.get_device_info_by_index(d_id)["defaultSampleRate"])

    
def main():
    device_id, device_rate = input_device()  
    plotwin=PlotWindow(device_id, device_rate)
    
    if (sys.flags.interactive!=1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    
if __name__=="__main__":
    main()
