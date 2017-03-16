#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os.path
import math
import json
import time
from datetime import datetime
from operator import itemgetter

import h5py
import tqdm
import copy

#calc
import numpy as np
from numpy import linalg as NLA
import scipy as sp
from scipy import linalg as SLA
from scipy.spatial import distance as DIST
from scipy import stats
from scipy import signal

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA as skCCA

#GUI
from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

#plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#ROS
import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA


import data_proc
import data_proc2


class Plot():
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        #pl.ion()
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(parent)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.rho_area = self.fig.add_subplot(111)
        #self.rho_area = self.fig.add_subplot(211)
        self.rho_area.set_title("rho", fontsize=11)

        self.fig.tight_layout()

        #ROS
        rospy.init_node('cor_joints', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

        #rvizのカラー設定(未)
        self.carray = []
        clist = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1]]
        for c in clist:
            color = ColorRGBA()
            color.r = c[0]
            color.g = c[1]
            color.b = c[2]
            color.a = c[3]
            self.carray.append(color)

        self.jidx = [[11, 3, 2, 10, 1, 0],
                     [10, 4, 5, 6],
                     [10, 7, 8, 9],]

    #プロットエリアがクリックされた時
    def on_click(self, event):
        print 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(
            event.button, event.x, event.y, event.xdata, event.ydata)
        row = int(event.ydata)
        col = int(event.xdata)
        print "---(",row,", ",col,")---"
        print 'cca r:',self.r_m[row][col]
        self.pubViz(row, col)
        
        
    def rviz_obj(self, obj_id, obj_ns, obj_type, obj_size, obj_color=0, obj_life=0):
        obj = Marker()
        obj.header.frame_id, obj.header.stamp = "camera_link", rospy.Time.now()
        obj.ns, obj.action, obj.type = str(obj_ns), 0, obj_type
        obj.scale.x, obj.scale.y, obj.scale.z = obj_size[0], obj_size[1], obj_size[2]
        obj.color = self.carray[obj_color]
        obj.lifetime = rospy.Duration.from_sec(obj_life)
        obj.pose.orientation.w = 1.0
        return obj

    def set_point(self, pos, addx=0, addy=0, addz=0, rotate=False):
        pt = Point()
        if rotate == True:
            pt.x, pt.y, pt.z = -1*pos[0]+addx, -1*pos[1]+addy, pos[2]+addz
        else:
            pt.x, pt.y, pt.z = pos[0]+addx, pos[1]+addy, pos[2]+addz
        return pt

    #def pubViz(self, t_user1, t_user2):
    #def pubViz(self, r, c, cor, wts, poses, wins, orgs):
    def pubViz(self, row, col):
        
        #print "r, c", row, col
        rate = rospy.Rate(17)
        #offset = c if  r>c  else r

        #datasize = len(self.data1)
        window = self.window
        offset = self.offset

        d1_start = col
        d1_end = d1_start+window
        viz_data1 = self.data1[d1_start:d1_end,:]#(34, 36)

        d2_start = col-(row-offset-1)
        d2_end = d2_start+window
        viz_data2 = self.data2[d2_start:d2_end,:]

        vel1=self.vel1[d1_start:d1_end]
        vel2=self.vel2[d2_start:d2_end]

        self.draw_raw_data(vel1, vel2)
        #plt.plot(vel1)
        #plt.plot(vel2)
        #plt.show()
        #plt.scatter(vel1, vel2)
        #plt.show()

        
        viz_poses = [viz_data1, viz_data2]
        viz_rotates = [True, False]
        viz_offsets = [0, 1]
        for i in range(len(viz_data1)):
            msgs = MarkerArray()
            """
            sq = i+offset
            print "frame:",sq

            tmsg = self.rviz_obj(10, 'f10', 9, [0.1, 0.1, 0.1], 0)
            tmsg.pose.position.x,tmsg.pose.position.y,tmsg.pose.position.z=0,0,0
            tmsg.text = "c:"+str(round(cor, 3))+", f:"+str(sq)
            msgs.markers.append(tmsg) 
            """
            #print i
            for u in range(2):
                # points
                pmsg = self.rviz_obj(u, 'p'+str(u), 7, [0.03, 0.03, 0.03], 0)
                #print np.array(viz_poses[u][i,:]).shape
                vpos = np.array(viz_poses[u][i,:]).reshape(12,3)#決めうち
                pmsg.points = [self.set_point(p, rotate=viz_rotates[u], addx=viz_offsets[u]) for p in vpos]
                msgs.markers.append(pmsg)

                
                # lines
                lmsg = self.rviz_obj(u, 'l'+str(u), 5, [0.005, 0.005, 0.005], 2)
                for jid in self.jidx:
                    for pi in range(len(jid)-1):
                        for add in range(2):
                            lmsg.points.append(self.set_point(vpos[jid[pi+add]],
                                                              rotate=viz_rotates[u],
                                                              addx=viz_offsets[u])) #vpos(12,3) 
                msgs.markers.append(lmsg)
                
                """
                # text
                tjs = 0.1
                tmsg = self.rviz_obj(u, 't'+str(u), 9, [tjs, tjs, tjs], 0)
                tmsg.pose.position = self.set_point(pos[i][3], addy=tjs, addz=tjs)
                tmsg.text = "user_"+str(u+1)
                msgs.markers.append(tmsg) 
                """      
            self.mpub.publish(msgs)
            rate.sleep()

    def draw_raw_data(self, data1, data2):

        #data1 = signal.savgol_filter(data1, self.window/4, 5)
        #data2 = signal.savgol_filter(data2, self.window/4, 5)
                    
        self.tmp_x_area.cla()
        self.tmp_x_area.plot(data1)
        self.tmp_x_area.plot(data2)
        #self.tmp_x_area.plot(data1)
        self.tmp_x_area.set_title("raw_data", fontsize=11)
        #self.tmp_x_area.set_ylim(-0.5, 0.5)
        #self.tmp_x_area.set_ylim(-0.6, 0.6)
        
        self.tmp_y_area.cla()
        self.tmp_y_area.scatter(data1, data2)
        #self.tmp_y_area.plot(data2)
        self.tmp_y_area.set_title("scatter", fontsize=11)

        self.tmp_y_area.set_xlim(-0.3, 0.3)
        self.tmp_x_area.set_ylim(-0.3, 0.3)
        
        self.fig.canvas.draw()
        self.fig.tight_layout() 
        
    def scale(self, X):
        data = (X - np.mean(X, axis=0))/np.std(X, axis=0)
        return data




    def on_draw(self, r, data1, data2, vel1, vel2, annos, times, window, offset):
        self.r_m = r
        self.data1 = data1
        self.data2 = data2
        self.vel1 = vel1
        self.vel2 = vel2
        self.window = window
        self.offset = offset


        #(offset*2+1, datasize-window)のr_mを(offset*2+1,)となるよう合計をとる
        print "r_m", self.r_m.shape


        
        #plt.show()





        
        #データの可視化
        #fig = plt.figure()
        #ax1 = plt.subplot2grid((3,4), (0,0), rowspan=2, colspan=3)
        #plt.subplot(2,1,1)
        ax1 = plt.subplot2grid((1,4), (0,0), colspan=3)
        fs = 10
        dr, dc = self.r_m.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]

        #print "min rho", np.min(self.r_m)
        #print "data shape",self.data1.shape,self.data2.shape

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #axp = ax.imshow(np.random.randint(0, 100, (100, 100)))
        # Adding the colorbar
        #cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])  # This is the position for the colorbar
        #cb = plt.colorbar(axp, cax = cbaxes)
        
        img = plt.pcolor(X, Y, self.r_m, vmin=-1.0, vmax=1.0, cmap=cm.bwr)
        #plt.colorbar(img)
        
        #plt.tick_params(labelsize=5)

        plt.xlim(0, dc)
        plt.ylim(0, dr)

 
        diff_times = []
        for i in range(len(times)):
            diff_times.append(times[i]-times[0])

        diff_times = np.round(diff_times,1)
        
        plt.xticks(np.arange(dc)[::500], diff_times[::500])
        #print len(np.arange(dc)[::500]), len(diff_times[::500])

        
        esp_times = []
        for i in range(len(times)-1):
            if i == 0:
                esp_times.append(0)
            esp_times.append(times[i+1]-times[i])
        esp_times = np.round(esp_times, 1)
        ave_esp_time = np.average(esp_times)
        print ave_esp_time, " fps:",1/ave_esp_time
        
        
        wid = 5
        ticks = np.arange(dr)[::wid]
        use_time = diff_times[:offset]
        labels = np.hstack((use_time[::-1],0,-use_time))[::wid]

        #print labels
        
        plt.yticks(ticks, labels)
        plt.xlabel("time[sec]")
        plt.ylabel("offset[sec]")
        title = "moving cross correlation"
        plt.title(title)


        """
        #plt.subplot(2,1,2)
        ax2 = plt.subplot2grid((3,4), (2,0), colspan=3)
        #ax2 = plt.subplot2grid((3,1), (2,0))
        data1 = annos[:,0]
        data2 = annos[:,1]
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

                
        plt.yticks([data2_y, data1_y],["robot","person"])
        plt.xticks(np.arange(dc)[::500], diff_times[::500])
        plt.xlim(0, dc)
        plt.ylim(-1, 2)
        plt.xlabel("time[sec]")
        plt.title("speaker/listener result")
        """




        #ax3 = plt.subplot2grid((3,4), (0,3), rowspan=2)
        ax3 = plt.subplot2grid((1,4), (0,3))
        abs_rm = np.fabs(self.r_m)
        sum_rm = np.sum(abs_rm, axis=1)
        datasize = self.r_m.shape[1]
        dev = np.ones(self.offset)*datasize
        dev = dev - np.arange(1,self.offset+1)
        dev_rm = np.hstack((dev[::-1],datasize,dev))
        ave_rm = sum_rm/dev_rm

        #locate = np.arange(self.offset*2+1)[::5]
        #locate = np.arange()
        
        #labels = np.hstack((np.arange(1,self.offset+1)[::-1],0,-np.arange(1,self.offset+1)))[::5]
        plt.plot(ave_rm, np.arange(len(ave_rm)))

        #print ','.join(map(str, ave_rm))
        
        #plt.yticks(locate, labels)
        plt.yticks(ticks, labels)

        print ','.join(map(str, ticks))
        print ','.join(map(str, labels))
        #timeline = np.hstack((use_time[::-1],0,-use_time))
        #print ','.join(map(str, timeline))
        
        plt.xlim(0, 0.5)
        #plt.ylim(0, dr)

        #print dr, 
        
        plt.ylabel("offset[sec]")
        plt.xlabel("abs correlation")
        plt.title("abs correlation average")
        plt.grid()
        plt.rcParams["font.size"] = 10
        
        plt.tight_layout()        
        plt.show()  
    


    

    
    def on_draw_tmp(self, r, data1, data2, vel1, vel2, window, offset):
        self.r_m = r
        self.data1 = data1
        self.data2 = data2
        self.vel1 = vel1
        self.vel2 = vel2
        self.window = window
        self.offset = offset



        
        #(offset*2+1, datasize-window)のr_mを(offset*2+1,)となるよう合計をとる
        print "r_m", self.r_m.shape

        """
        for rows in range(self.r_m.shape[0]):
            for cols in range(self.r_m.shape[1]):
                self.r_m[rows][cols] = 0 if self.r_m[rows][cols]<0 else self.r_m[rows][cols]
        """
        
        abs_rm = np.fabs(self.r_m)
        sum_rm = np.sum(abs_rm, axis=1)
        datasize = self.r_m.shape[1]
        dev = np.ones(self.offset)*datasize
        dev = dev - np.arange(1,self.offset+1)
        dev_rm = np.hstack((dev[::-1],datasize,dev))
        ave_rm = sum_rm/dev_rm

        locate = np.arange(self.offset*2+1)[::5]
        #locate = np.arange()
        
        labels = np.hstack((np.arange(1,self.offset+1)[::-1],0,-np.arange(1,self.offset+1)))[::5]
        plt.plot(ave_rm)
        plt.xticks(locate, labels)
        plt.ylim(0,0.3)
        
        plt.show()




        
        self.rho_area = self.fig.add_subplot(111)
        #self.rho_area = self.fig.add_subplot(211)

        #データの可視化
        fs = 10
        dr, dc = self.r_m.shape
        Y, X = np.mgrid[slice(0, dr+1, 1),slice(0, dc+1, 1)]

        print "min rho", np.min(self.r_m)
        print "data shape",self.data1.shape,self.data2.shape
        
        if np.min(self.r_m) < 0:    
            img = self.rho_area.pcolor(X, Y, self.r_m, vmin=-1.0, vmax=1.0, cmap=cm.bwr)#gray #bwr
        else:
            img = self.rho_area.pcolor(X, Y, self.r_m, vmin=0.0, vmax=1.0, cmap=cm.gray)#gray #bwr
        """
        if self.cbar == None:
            self.cbar = self.fig.colorbar(img)
            self.cbar.ax.tick_params(labelsize=fs-1) 
        """
        self.rho_area.set_xlim(0, dc)
        self.rho_area.set_ylim(0, dr)

        wid = 10 #とりあえず決め打ちで10ずつ目盛表示
        ticks = [i*wid for i in range(dr/wid+1)]
        labels = [(dr-1)/2-i*wid for i in range(dr/wid+1)]
        self.rho_area.set_yticks(ticks=ticks)
        self.rho_area.set_yticklabels(labels=labels)
        self.rho_area.set_xlabel("user 1")
        self.rho_area.set_ylabel("user 2")

        self.rho_area.tick_params(labelsize=fs)
        self.rho_area.set_title("rho", fontsize=fs+1)
        self.fig.canvas.draw()

"""
self.rho_area = self.fig.add_subplot(211)
self.rho_area.plot()
self.rho_area.set_xlim(0, dc)
self.rho_area.set_ylim(0, dr)
"""

class CCA(QtGui.QWidget):

    def __init__(self):
        super(CCA, self).__init__()
        #UI
        self.init_ui()
        

    def init_ui(self):

        #Botton Objectの作成
        def boxBtnObj(name, func, maxlen=30):
            box = QtGui.QHBoxLayout()
            btn = btnObj(name, func, maxlen=maxlen)
            box.addWidget(btn)
            return box

        def btnObj(name, func, maxlen=30):
            btn = QtGui.QPushButton(name)
            btn.setMaximumWidth(maxlen)
            btn.clicked.connect(func)
            return btn


        
        grid = QtGui.QGridLayout()
        form = QtGui.QFormLayout()

        # 使用するDataの長さ
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('-1')
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('data size', self.frmSizeBox)
        
        
        # 使用するFileの指定
        self.txtSepFile = QtGui.QLineEdit()
        btnSepFile = btnObj("...", self.chooseDbFile, maxlen=40)
        btnRawInput = btnObj("raw", self.inputData, maxlen=60)
        btnProcedInput = btnObj("proced", self.inputProcedData, maxlen=60)
        
        boxSepFile = QtGui.QHBoxLayout()
        boxSepFile.addWidget(self.txtSepFile)
        boxSepFile.addWidget(btnSepFile)
        boxSepFile.addWidget(btnRawInput)
        boxSepFile.addWidget(btnProcedInput)
        form.addRow('input', boxSepFile)

        """
        # 使うdataの始まり/終わりを指定
        self.dataStart = QtGui.QLineEdit()
        self.dataStart.setText('0')
        self.dataStart.setFixedWidth(70)
        self.dataEnd = QtGui.QLineEdit()
        self.dataEnd.setText('500')
        self.dataEnd.setFixedWidth(70)
        boxDatas = QtGui.QHBoxLayout()
        boxDatas.addWidget(self.dataStart)
        boxDatas.addWidget(self.dataEnd)
        form.addRow('data range', boxDatas)
        """
      
        #window size
        self.winSizeBox = QtGui.QLineEdit()
        self.winSizeBox.setText('34')
        self.winSizeBox.setAlignment(QtCore.Qt.AlignRight)
        self.winSizeBox.setFixedWidth(100)
        form.addRow('window size', self.winSizeBox)

        
        #どれだけずらすか offset size
        self.offsetBox = QtGui.QLineEdit()
        self.offsetBox.setText('10')
        self.offsetBox.setAlignment(QtCore.Qt.AlignRight)
        self.offsetBox.setFixedWidth(100)
        form.addRow('offset frames', self.offsetBox)

        
        #pick up
        self.pickUpBox = QtGui.QLineEdit()
        self.pickUpBox.setText('1')
        self.pickUpBox.setAlignment(QtCore.Qt.AlignRight)
        self.pickUpBox.setFixedWidth(100)
        form.addRow('pick up', self.pickUpBox)
        
        """
        # regulation
        self.regBox = QtGui.QLineEdit()
        self.regBox.setText('0.0')
        self.regBox.setAlignment(QtCore.Qt.AlignRight)
        self.regBox.setFixedWidth(100)
        form.addRow('regulation', self.regBox)
        """
        
        # threshold
        self.thBox = QtGui.QLineEdit()
        self.thBox.setText('0.0')
        self.thBox.setAlignment(QtCore.Qt.AlignRight)
        self.thBox.setFixedWidth(100)
        form.addRow('threshold', self.thBox)

        """
        rHLayout = QtGui.QHBoxLayout()
        self.radios = QtGui.QButtonGroup()
        self.allSlt = QtGui.QRadioButton('all frame')
        self.radios.addButton(self.allSlt)
        rHLayout.addWidget(self.allSlt)
        form.addRow('select', rHLayout)
        """
        
        """
        #progress bar
        self.pBar = QtGui.QProgressBar()
        form.addRow('progress', self.pBar)
        """

        #exec button
        boxCtrl = QtGui.QHBoxLayout()
        btnExec = QtGui.QPushButton('exec')
        btnExec.clicked.connect(self.do_exec)
        #btnExec.clicked.connect(self.manyFileExec)
        boxCtrl.addWidget(btnExec)

        #output file
        boxPlot = QtGui.QHBoxLayout()
        btnPlot = QtGui.QPushButton('plot')
        btnPlot.clicked.connect(self.rhoplot)
        boxPlot.addWidget(btnPlot)

        #output file
        boxFile = QtGui.QHBoxLayout()
        btnOutput = QtGui.QPushButton('output')
        #btnOutput.clicked.connect(self.save_params)
        boxFile.addWidget(btnOutput)

        # matplotlib
        boxPlot = QtGui.QHBoxLayout()
        self.main_frame = QtGui.QWidget()
        self.plot = Plot(self.main_frame)
        boxPlot.addWidget(self.plot.canvas)

        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxCtrl,2,0)
        grid.addLayout(boxFile,3,0)
        grid.addLayout(boxPlot,4,0)

        self.setLayout(grid)
        #self.resize(400,100)

        self.setWindowTitle("cca window")
        self.show()

    def chooseDbFile(self):
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.ExistingFile)
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for f in fileNames:
                self.txtSepFile.setText(f)
                return
        return self.txtSepFile.setText('')

    def updateColorTable(self, cItem):
        self.r = cItem.row()
        self.c = cItem.column()
        print "now viz r:",self.r,", c:",self.c


    # 生Dataの入力
    def inputData(self):
        filesize = int(self.frmSizeBox.text())
        print "Input raw data:", filesize
        self.fname = [str(self.txtSepFile.text())]
        input_data = data_proc2.load_persons(self.fname, annobool=True,
                                             datalen=filesize)
        self.loadInputData(input_data)


    # 加工済のDataの入力  
    def inputProcedData(self):
        filesize = int(self.frmSizeBox.text())
        print "Input proced data", filesize
        self.fname = [str(self.txtSepFile.text())]
        input_data = data_proc2.load_proced_data_flag(self.fname, datalen=filesize)
        
        self.loadInputData(input_data)
        

    def loadInputData(self, input_data):
        self.input_joints = input_data["joints"]
        self.input_speaks = input_data["speaks"]
        datalen=self.input_speaks.shape[0]

        if input_data.has_key("annos"):
            self.input_annos = input_data["annos"]
        else:
            self.input_annos = np.zeros((datalen, self.anno_dim))#(1000,2)
            
        if input_data.has_key("flags"):
            print "load flags"
            self.input_flags = input_data["flags"]
        else:
            print "create flags"
            self.input_flags = np.zeros((datalen, 1))

        if input_data.has_key("times"):
            print "load times"
            self.input_times = input_data["times"]
        else:
            print "create times"
            self.input_times = np.zeros((datalen, 1))

        
        print "joints shape:", self.input_joints.shape
        print "speaks shape:", self.input_speaks.shape
        print "annos shape:", self.input_annos.shape
        print "times shape:", self.input_times.shape

        self.edited_joints = copy.deepcopy(self.input_joints)#np.arrayでCopyされる
        self.edited_speaks = copy.deepcopy(self.input_speaks)#speakは編集しない(今のところ)
        self.edited_annos = copy.deepcopy(self.input_annos)
        self.edited_times = copy.deepcopy(self.input_times)

        #平滑化してみる, 範囲35
        #self.edited_joints = self.smoothing(self.edited_joints)
        
    #PCAを用いた次元削減
    #u1.shape:(1000, 36) -> (1000, 5)
    def dimension_reduction(self, data):

        def pca_transform(X, idx):
            pca = PCA(n_components=1)
            X = X[:,idx]
            pca.fit(X)
            res = pca.transform(X)

            #res = th_mean(res)

            
            #plt.subplot(2,1,1)
            #plt.plot(X)
            #plt.subplot(2,1,2)
            #plt.plot(res)
            #plt.show()
            
            
            return res

        # X.shape (1000,1)
        def standerd(X):
            ave = np.sum(X)/len(X)
            std = np.std(X)
            return (X-ave)/std

        # Dataがある一定以上動かない場合(微妙に動くがたまたまその値が強い相関をとってしまう)
        # Dataを平均値に変換する
        def th_mean(X):
            std = np.std(X)
            print std
            if std < 0.03:
                ave = np.sum(X)/len(X)
                res = np.ones(X.shape)*ave
                return  res
            return X
        
        dim = data.shape[1]/2
        users = [data[:,:dim], data[:,dim:]]#(1000, 36)*2

        red_users = []
        for user in users:

            scale = 100
            
            idx = [2*3,2*3+1,2*3+2,3*3,3*3+1,3*3+2,11*3,11*3+1,11*3+2]
            head = pca_transform(user, idx)
            
            idx = [5*3,5*3+1,5*3+2,6*3,6*3+1,6*3+2]
            l_arm = pca_transform(user,idx)

            idx = [8*3,8*3+1,8*3+2,9*3,9*3+1,9*3+2]
            r_arm = pca_transform(user,idx)

            idx = [4*3,4*3+1,4*3+2,7*3,7*3+1,7*3+2]
            shoulder = pca_transform(user,idx)

            idx = [0*3,0*3+1,0*3+2,1*3,1*3+1,1*3+2,10*3,10*3+1,10*3+2]
            spine = pca_transform(user,idx)
            
            """
            plt.subplot(5,1,1)
            plt.plot(head)
            plt.subplot(5,1,2)
            plt.plot(l_arm)
            plt.subplot(5,1,3)
            plt.plot(r_arm)
            plt.subplot(5,1,4)
            plt.plot(shoulder)
            plt.subplot(5,1,5)
            plt.plot(spine)
            plt.show()
            """
            red = np.hstack((head, l_arm, r_arm, shoulder, spine))
            print red.shape
            red_users.append(red)
            
        return red_users[0], red_users[1]
        #print "no pca"
        #return users[0], users[1]


    def conv_velocity(self, data):

        #data(1000, 36)
        vels = []
        for i in range(len(data)):
            if i == 0:
                vel = 0
            else:
                vel_v = data[i]-data[i-1]
                vel = np.sqrt(np.dot(vel_v, vel_v.T))

            vels.append(vel)
        
        return np.array(vels)

    
    def conv_select_velocity(self, data, idx):

        #data(1000, 36)
        vels = []
        for i in range(len(data)):
            if i == 0:
                vel = 0
            else:
                vel_v = data[i][idx]-data[i-1][idx]
                vel = np.sqrt(np.dot(vel_v, vel_v.T))

            vels.append(vel)
        
        return np.array(vels)

    
    #data(1000, 72)
    def smoothing(self, data):
        print data.shape
        sm_data = []
        for seq in data.T:
            #print seq.shape
            #print seq
            ren_window = 35#self.window/4+1 if self.window%2==0 else self.window/4 
            sm = signal.savgol_filter(seq, ren_window, 3)
            sm_data.append(sm)
            
        return np.array(sm_data).T

        
        #self.data2 = signal.savgol_filter(self.data2, self.window/4, 5)
        
    
    def do_exec(self):
        print "exec start:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        dim = 36
        
        #ws:window_size, fs:frame_size 
        self.window = int(self.winSizeBox.text())
        self.offset = int(self.offsetBox.text())
        self.pickup = int(self.pickUpBox.text())
        
        # Data前処理 self.edited_joints #(1000, 72)
        #self.data1, self.data2 = self.dimension_reduction(self.edited_joints)

        src_data1, src_data2 = self.edited_joints[::self.pickup,:dim], self.edited_joints[::self.pickup,dim:]
        self.edited_annos = self.edited_annos[::self.pickup,:]
        self.edited_times = self.edited_times[::self.pickup]
        
        #src_data1, src_data2 = self.smoothing(src_data1), self.smoothing(src_data2) 
        
        self.data1, self.data2 = src_data1, src_data2 #self.edited_joints[:,:dim], self.edited_joints[:,dim:]
        
        # 可視化用        
        self.viz_data1, self.viz_data2 = src_data1, src_data2 #self.edited_joints[:,:dim], self.edited_joints[:,dim:]

        #plt.plot(self.data2)
        #plt.show()
         
        #異なるData用
        #self.data1, self.data2 = self.edited_joints[:3000,:dim], self.edited_joints[5000:8000,dim:]
        #self.viz_data1, self.viz_data2 = self.edited_joints[:3000,:dim], self.edited_joints[5000:8000,dim:]
        
        # data1:(1000,), data2:(1000,)
        self.data1, self.data2 = self.conv_velocity(self.data1), self.conv_velocity(self.data2)


        ymax = 0.4
        xmax, xmin = 2000, 300
        pdata1, pdata2 = self.data1[xmin:xmax], self.data2[xmin:xmax]
        plt.subplot(2,1,1)
        plt.plot(pdata1, "r")
        plt.xlim(0, len(pdata1))
        #plt.xlim(xmin, xmax)
        plt.ylim(0, ymax)
        plt.title("person")
        plt.subplot(2,1,2)
        plt.plot(pdata2, "g")
        plt.xlim(0, len(pdata2))
        #plt.xlim(xmin, xmax)
        plt.ylim(0, ymax)
        plt.title("robot")
        plt.show()


        

        self.data_size  = len(self.data1)
        self.dim1 = 1
        self.dim2 = 1 #self.data2.shape[1]
    

        #dtmr:data_max_range,
        self.max_range = self.data_size - self.window + 1

        print "datas_size:",self.data_size
        print "frame_size:",self.offset
        print "data_max_range:",self.max_range

        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        #self.reg = float(self.regBox.text())
        self.th = float(self.thBox.text())
        #self.r_m, self.wx_m, self.wy_m, self.js1, self.js2 = self.cca_exec(self.data1, self.data2)
        self.r_m = self.cor_exec(self.data1, self.data2)


        #self.r_m, self.tmp_x, self.tmp_y = self.cca_exec(self.data1, self.data2)

        #graph
        self.rhoplot()
        
        
        print "end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    def rhoplot(self):
        #self.plot.on_draw(self.r_m[:,:,0])
        self.plot.on_draw(self.r_m[:,:], self.viz_data1, self.viz_data2,
                          self.data1, self.data2, self.edited_annos, self.edited_times,
                          self.window, self.offset)
        



    def json_time_input(self, filename, start, end):
        f = open(filename, 'r')
        jD = json.load(f)
        f.close()
        times = []
        if jD[0]["datas"][0].has_key("time"): 
            for tobj in jD[0]["datas"]:
                times.append(tobj["time"])
            times = times[start:end]
        else:
            print "[WARN] no time data!"
        return times        

    """
    def save_params(self):
        
        save_dimen=1 #self.dtd
        savefile = "save_"+ self.fname.lstrip("/home/uema/catkin_ws/src/pose_cca/datas/")
        savefile = savefile.rstrip(".json")+"_w"+str(self.wins)+"_f"+str(self.frms) +"_d"+str(self.dtd)+"_r"+str(self.reg)+"_t"+str(self.th)+"_s"+str(self.start)+"_e"+str(self.end) 
        filepath = savefile+".h5"
        print filepath+" is save"
        with h5py.File(filepath, 'w') as f:
            p_grp=f.create_group("prop")
            p_grp.create_dataset("wins",data=self.window)
            p_grp.create_dataset("frms",data=self.offset)
            
            p_grp.create_dataset("dim1",data=self.dim1)
            p_grp.create_dataset("dim2",data=self.dim2)
            
            p_grp.create_dataset("pre_dtd",data=self.pre_dtd)
            p_grp.create_dataset("dts",data=self.dts) 
            #p_grp.create_dataset("fname",data=self.fname)
            p_grp.create_dataset("sidx",data=self.sIdx)
            p_grp.create_dataset("reg",data=self.reg)
            p_grp.create_dataset("th",data=self.th)
            p_grp.create_dataset("org1",data=self.org1)
            p_grp.create_dataset("org2",data=self.org2)
            c_grp=f.create_group("cca")
            c_grp.create_dataset("times", data=self.times)

            #変形したデータ
            d_grp=c_grp.create_group("data")
            d_grp.create_dataset("data1", data=self.data1)
            d_grp.create_dataset("data2", data=self.data2)

            #生データ
            rd_grp=c_grp.create_group("raw_data")
            rd_grp.create_dataset("raw_data1",data=self.raw_data1)
            rd_grp.create_dataset("raw_data2",data=self.raw_data2)

            r_grp=c_grp.create_group("r")
            wx_grp=c_grp.create_group("wx")
            wy_grp=c_grp.create_group("wy")

            #print "now save only r_m"
            #r_grp.create_dataset(str(0),data=self.r_m[:,:])

            #save_dimen = 1
            for i in xrange(save_dimen):
                r_grp.create_dataset(str(i),data=self.r_m[:,:])
                wx_v_grp = wx_grp.create_group(str(i))
                wy_v_grp = wy_grp.create_group(str(i))
                for j in xrange(self.dtd):
                    #print len(self.wx_m)
                    #print self.wx_m
                    wx_v_grp.create_dataset(str(j),data=self.wx_m[:,:,j])
                    wy_v_grp.create_dataset(str(j),data=self.wy_m[:,:,j])
            f.flush()
        print "save end:",datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    """


    def conv_th_means(self, data1, data2):
        th = 0.03
        std1 = np.std(data1)
        std2 = np.std(data2)

        if std1 < th and std2 < th:
            #print std1, std2
            if std1 > std2:
                ave = np.sum(data2)/len(data2)
                data2 = np.ones(data2.shape)*ave
            else:
                ave = np.sum(data1)/len(data1)
                data1 = np.ones(data1.shape)*ave
        elif std1 < th:
            ave = np.sum(data1)/len(data1)
            data1 = np.ones(data1.shape)*ave
        elif std2 < th:
            ave = np.sum(data2)/len(data2)
            data2 = np.ones(data2.shape)*ave
            
        return data1, data2



    

    def cor_exec(self, data1, data2):
        
        r_m = np.zeros([self.offset*2+1, self.max_range])
        for i in tqdm.tqdm(xrange(self.max_range)):
            for j in xrange(self.offset*2+1):
                if self.offset+i-j >= 0 and self.offset+i-j < self.max_range:
                    u1 = data1[i:i+self.window]                
                    u2 = data2[self.offset+i-j:self.offset+i-j+self.window]

                    #u1 = signal.savgol_filter(u1, self.window/4, 5)
                    #u2 = signal.savgol_filter(u2, self.window/4, 5)
                    
                    #u1, u2 = self.conv_th_means(u1, u2)
                    
                    r = np.corrcoef(u1.T, u2.T)[0,1]

                    r_m[j][i] = r if np.fabs(r)>self.th else 0
                        

        return r_m


    
    def cca_exec(self, data1, data2):
        #rho_m:rho_matrix[dmr, dmr, datadimen] is corrs
        #wx_m and wy_m is vectors
        data1 = np.array(data1)
        data2 = np.array(data2)

        r_m = np.zeros([self.offset*2+1, self.max_range])
        wx_m = np.zeros([self.offset*2+1, self.max_range, self.dim1])
        wy_m = np.zeros([self.offset*2+1, self.max_range, self.dim2])

        # これ、参照するときおかしくなってないか
        js1 = [[[[]] for i in range(self.max_range)] for j in range(self.offset*2+1)]
        js2 = [[[[]] for i in range(self.max_range)] for j in range(self.offset*2+1)]
        #
        #print js1
        
        #th = 0.3
        #row->colの順番で回したほうが効率いい

        #plt.ion()
        slopes = []
        count = 0
        od = 3
        for i in tqdm.tqdm(xrange(self.max_range)):
            for j in xrange(self.offset*2+1):
                if self.offset+i-j >= 0 and self.offset+i-j < self.max_range:
                    u1 = data1[i:i+self.window, :]                
                    u2 = data2[self.offset+i-j:self.offset+i-j+self.window,:]

                    np.corrcoef()
                    
                    
                    #j1, j2 = self.order_std_cut_joints(u1, od, self.th), self.order_std_cut_joints(u2, od, self.th)

                    # 使わないdimを選択している?
                    j1, j2 = self.std_cut_joints(u1, self.th), self.std_cut_joints(u2, self.th)


                    #u1, u2 = self.th_mean(u1), self.th_mean(u2)

                    
                    
                    #plt.plot(u1, "r")
                    #plt.plot(u2, "b")
                    #plt.show()
                    #print self.dim1, self.dim2
                    #j1, j2 = np.arange(0, self.dim1, 1).tolist(), np.arange(0, self.dim2, 1).tolist()
                    #print self.dim1, self.dim2
                    #print j1
                    #print j2
                    #j1, j2 = self.ex_joints(u1, self.th), self.ex_joints(u2, self.th)
                    #j1, j2 = self.diff_num_joints(u1, self.th), self.diff_num_joints(u2, self.th)
                    #print u1[:,j1]

                    
                    if len(j1) < 1 or len(j2) < 1:
                        # 使える次元がどちらか1以下なら計算不能
                        r_m[j][i] = 0
                    else:
                        r_m[j][i], wx_m[j][i], wy_m[j][i] = self.skcca(u1, u2, j1, j2)
                        #print "cca:", r_m[j][i] 
                        # 使ったJointをjson形式で保存
                        js1[j][i], js2[j][i] = json.dumps(j1), json.dumps(j2)


                        wx_arg = np.argmax(np.fabs(wx_m[j][i]))
                        select_u1 = u1[:,wx_arg]
                        
                        wy_arg = np.argmax(np.fabs(wy_m[j][i]))
                        select_u2 = u2[:,wy_arg]

                        slp, ipt, rv, _, _ = stats.linregress(select_u1,select_u2)


                        r_m[j][i] = self.tri_function(slp)*r_m[j][i]

                        
                        if np.fabs(slp) > 0.5 and np.fabs(slp) < 2:
                            #print j, i, r_m[j][i], wx_arg, wy_arg
                            
                            #plt.scatter(select_u1,select_u2)
                            #plt.xlim(-0.05,0.05)
                            #plt.ylim(-0.05,0.05)
                            
                            slopes.append(slp)
                        
                        count += 1
                        """
                        if r_m[j][i] > 0.997:
                           
                        # もっとも影響をあたえた次元同士の散布図をみたい
                            wx_arg = np.argmax(np.fabs(wx_m[j][i]))
                            select_u1 = u1[:,wx_arg]
                            
                            wy_arg = np.argmax(np.fabs(wy_m[j][i]))
                            select_u2 = u2[:,wy_arg]
                            
                            print j, i, r_m[j][i], wx_arg, wy_arg
                            
                            plt.scatter(select_u1,select_u2)
                            plt.xlim(-0.05,0.05)
                            plt.ylim(-0.05,0.05)
                            
                            
                        """
                        #print "slope, corr:", slp, rv

                        """
                        select_corr = np.corrcoef(select_u1.T, select_u2.T)
                        #print select_corr
                        r_m[j][i] = select_corr[0,1]

                        # 絶対値
                        # r_m[j][i] = np.fabs(r_m[j][i])


                        if r_m[j][i] > 0.9:
                            print j, i, r_m[j][i], wx_arg, wy_arg
                            wx, wy = wx_m[j][i], wy_m[j][i]
                            for jidx, u in enumerate(u1.T):
                                lb = str(jidx)+":"+str(round(wx[jidx],4))
                                plt.plot(u, "b", label=lb)
                                
                            for jidx, u in enumerate(u2.T):
                                lb = str(jidx)+":"+str(round(wy[jidx],4))
                                plt.plot(u, "b", label=lb)

                            plt.plot(select_u1, "r")
                            plt.plot(select_u2, "r")
                            #plt.plot(u2)

                            plt.legend() 
                            plt.show()
                        """
                        """
                        if r_m[j][i] > 0.9:
                            wx, wy = wx_m[j][i], wy_m[j][i]
                            for jidx, u in enumerate(u1.T):
                                lb = str(jidx)+":"+str(round(wx[jidx],4))
                                plt.plot(u, label=lb)
                                
                            for jidx, u in enumerate(u2.T):
                                lb = str(jidx)+":"+str(round(wy[jidx],4))
                                plt.plot(u, "--", label=lb)
                                
                            #plt.plot(u2)

                            plt.legend() 
                            plt.show()
                        """
                    #r_m[j][i], wx_m[j][i], wy_m[j][i] = self.cca(u1, u2, j1, j2)
        #plt.show()
        #plt.plot(slopes)
        #plt.show()
        print "slope:",len(slopes),"/count:",count,"=",len(slopes)/float(count)
        print "min-max:", min(slopes), "-", max(slopes)
        return r_m, wx_m, wy_m, js1, js2


    
    def corr(self, X, Y):
        #print X.shape, Y.shape # ex:(34, 5)
        data1 = np.fabs(X).mean(axis=1)
        data2 = np.fabs(Y).mean(axis=1)

        return np.corrcoef(data1.T, data2.T)[0][1]



    def skcca(self, X, Y, j1, j2):

        X = X[:, j1]
        Y = Y[:, j2]
        
        cca = skCCA(n_components=1)
        #cca.fit(X, Y)
        X_c, Y_c = cca.fit_transform(X, Y)


        #plt.plot(X_c)
        #plt.plot(Y_c)
        #plt.show()
        
        res = np.corrcoef(X_c.T, Y_c.T)[0,1]
        Wx = np.zeros([self.dim1])
        Wy = np.zeros([self.dim2])

        Wx[j1] = cca.x_weights_
        Wy[j2] = cca.y_weights_
        
        return res, Wx, Wy
    
    
    def cca(self, X, Y, j1, j2):
        # ref: https://gist.github.com/satomacoto/5329310
        '''
        正準相関分析
        http://en.wikipedia.org/wiki/Canonical_correlation
        '''    
        X = X[:, j1]
        Y = Y[:, j2]

        #print X.shape, Y.shape
        
        n, p = X.shape
        n, q = Y.shape
                  
        # zero mean
        X = X - X.mean(axis=0)
        Y = Y - Y.mean(axis=0)
        
        # covariances
        S = np.cov(X.T, Y.T, bias=1)
        
        SXX = S[:p,:p]
        SYY = S[p:,p:]
        SXY = S[:p,p:]

        #正則化
        #SXX = self.add_reg(SXX, self.reg) 
        #SYY = self.add_reg(SYY, self.reg)

        sqx = SLA.sqrtm(SLA.inv(SXX)) # SXX^(-1/2)
        sqy = SLA.sqrtm(SLA.inv(SYY)) # SYY^(-1/2)
        M = np.dot(np.dot(sqx, SXY), sqy.T) # SXX^(-1/2) * SXY * SYY^(-T/2)
        A, r, Bh = SLA.svd(M, full_matrices=False)
        B = Bh.T     

        # 基底変換
        #A = np.dot(sqx, A)
        #B = np.dot(sqy, B)

        # 正準相関で変換した値
        U = np.dot(np.dot(A[:,0].T, sqx), X.T).T
        V = np.dot(np.dot(B[:,0].T, sqx), Y.T).T
        
        
        plt.plot(U)
        plt.plot(V)
        plt.show()

        
        Wx = np.zeros([self.dim1])
        Wy = np.zeros([self.dim2])

        Wx[j1] = A[:,0]
        Wy[j2] = B[:,0]
        
        return r[0], Wx, Wy

    def add_reg(self, reg_cov, reg):
        reg_cov += reg * np.average(np.diag(reg_cov)) * np.identity(reg_cov.shape[0])
        return reg_cov



def main():
    app = QtGui.QApplication(sys.argv)
    corr = CCA()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
