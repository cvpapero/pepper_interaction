#!/usr/bin/python
# -*- coding: utf-8 -*-

 
"""
2017.1.24---
時間


2017.1.4----
音声を生で扱う
Annotationとは別で扱う

2016.12.12----
20161210_proc3p.jsonを使う。反転させてみる

2016.12.9----
三人の対話におけるannotationのInterface
手動でannotationする
slider barとtableを同期させた

2016.10.19----
interaction dataをannotationするためのinterface

"""

import sys
import os.path
import math
import json
import time
import copy

import numpy as np

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui  import *

import matplotlib.pyplot as plt

import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA

import data_proc2

from std_msgs.msg import Float64MultiArray


class ANNOTATION(QtGui.QWidget):

    def __init__(self):
        super(ANNOTATION, self).__init__()
        #UIの初期化
        self.initUI()
 
        #ROSのパブリッシャなどの初期化
        rospy.init_node('annotation_interface2', anonymous=True)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
        #self.ppub = rospy.Publisher('joint_diff', PointStamped, queue_size=10)
        self.speakpub = rospy.Publisher('/speaks', Float64MultiArray, queue_size=10)

        #rvizのカラー設定(未)
        self.carray = []
        clist = [[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1]]
        for c in clist:
            color = ColorRGBA()
            color.r, color.g, color.b, color.a = c[0], c[1], c[2], c[3]
            self.carray.append(color)

        # set extra data param
        self.dim_x = 72
        self.llist = [[0, 1, 10, 2, 3, 11], [10, 4, 5, 6], [10, 7, 8, 9]]

        self.input_joints = []
        self.input_speaks = []
        self.anno_dim = 2
        
        self.r, self.c = 0, 0

        
            
    def initUI(self):

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

        #frame size
        self.frmSizeBox = QtGui.QLineEdit()
        self.frmSizeBox.setText('-1')
        self.frmSizeBox.setFixedWidth(100)
        form.addRow('size', self.frmSizeBox)
        
        #ファイル入力ボックス  
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
 

        #ファイル出力ボックス
        self.txtOutputFile = QtGui.QLineEdit()
        self.txtOutputFile.setText('test_proc.json')
        btnOutputFile = btnObj("save", self.outputData, maxlen=100)
        
        boxOutputFile = QtGui.QHBoxLayout()
        boxOutputFile.addWidget(self.txtOutputFile)
        boxOutputFile.addWidget(btnOutputFile)
        form.addRow('output', boxOutputFile)

        #data cut
        cutRange = QtGui.QHBoxLayout()
        self.cutStart = QtGui.QLineEdit()
        self.cutStart.setText('0')
        self.cutEnd = QtGui.QLineEdit()
        self.cutEnd.setText('0')
        btnCutRange = btnObj("exec", self.cutRangeData, maxlen=60)
        cutRange.addWidget(self.cutStart)
        cutRange.addWidget(self.cutEnd)
        cutRange.addWidget(btnCutRange)
        form.addRow('trimming', cutRange)
        
        
        # check range
        checkRange = QtGui.QHBoxLayout()
        self.checkUser = QtGui.QLineEdit()
        self.checkUser.setText('0')
        self.checkSt = QtGui.QLineEdit()
        self.checkSt.setText('0')
        self.checkEd = QtGui.QLineEdit()
        self.checkEd.setText('0')
        btnRangeCheck = btnObj("check", self.checkRangeData, maxlen=60)
        checkRange.addWidget(self.checkUser)
        checkRange.addWidget(self.checkSt)
        checkRange.addWidget(self.checkEd)
        checkRange.addWidget(btnRangeCheck)
        form.addRow('U/S/E', checkRange)

        
        # direct
        boxDirect = boxBtnObj("d", self.directJoints, maxlen=20)
        form.addRow('direct', boxDirect)
 
        
        # Reset
        boxResetAnno = boxBtnObj("e", self.resetAnno, maxlen=20)
        form.addRow('reset', boxResetAnno)

        # Reverse
        boxReverse = boxBtnObj("e", self.reverseData, maxlen=20)
        form.addRow('reverse', boxReverse)

        # Time Line
        boxSld = QtGui.QHBoxLayout()
        lcd = QtGui.QLCDNumber(self)
        self.sld = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.sld.valueChanged.connect(lcd.display)
        self.sld.valueChanged.connect(self.sliderChange)
        boxSld.addWidget(lcd)
        boxSld.addWidget(self.sld)
        
        
        #テーブルの初期化
        #horizonはuser2の時間
        self.table = QtGui.QTableWidget(self)
        self.table.setColumnCount(0)
        #self.table.setHorizontalHeaderLabels("use_2 time") 
        jItem = QtGui.QTableWidgetItem(str(0))
        self.table.setHorizontalHeaderItem(0, jItem)

        #アイテムがクリックされたらグラフを更新
        self.table.itemClicked.connect(self.clickUpdateTable)
        #self.table.itemActivated.connect(self.activatedUpdateTable)
        self.table.setItem(0, 0, QtGui.QTableWidgetItem(1))

        #self.itemSelectionChanged.connect(self.selection_changed)
        #self.tableSlider = self.table.verticalScrollBar()

        
        boxTable = QtGui.QHBoxLayout()
        boxTable.addWidget(self.table)
        
        
        #配置
        grid.addLayout(form,1,0)
        grid.addLayout(boxSld,2,0)
        grid.addLayout(boxTable,3,0)

        self.setLayout(grid)
        self.resize(400,100)

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

    # reset
    def resetAnno(self):
        print "reset Data before:",self.edited_joints.shape, self.edited_annos.shape
        self.edited_joints = copy.deepcopy(self.input_joints)
        self.edited_speaks = copy.deepcopy(self.input_speaks)
        #datalen = self.edited_speaks.shape[1]
        self.edited_annos = copy.deepcopy(self.input_annos) #np.zeros((datalen, self.anno_dim)) #(1000, 2)
        self.edited_flags = copy.deepcopy(self.input_flags)
        self.sld.setMaximum(self.edited_speaks.shape[0]-1) #
        
        self.updateTable(self.edited_speaks, self.edited_annos, self.edited_times)
        
        print "reset Data after:",self.edited_joints.shape, self.edited_speaks.shape, self.edited_annos.shape

    def speakNorm(self, data, upper=300, lower=80):
        # 上限
        c_data=data if data <upper else upper
        # 下限
        c_data=c_data if c_data>lower else lower
        # 最大値で割る
        return (c_data-lower)/float(upper-lower)

    def convDecibel(self, data):
        return 20*math.log10(data)
    #decibel
    def speakDecibelNorm(self, data):
        decibel = self.convDecibel(data)
        return self.speakNorm(decibel, upper=70, lower=30)

    
    def updateTable(self, data, anno, times):

        #plt.plot(data)
        #plt.show()
        
        th = 0#float(self.ThesholdBox.text())
        if(len(data)==0):
            print "No Data! Push exec button..."

        d_row, d_col = data.shape #(1000, 2)
        
        add_ann = anno.shape[1] #annotationの個数
        #print add_ann

        add_flag = 1
        #t_row:時間軸, t_col:次元数, add_ann:加えるAnnotation
        t_col = d_col+add_ann+add_flag
        t_row = d_row

        #print "t, d, a",t_col,d_col,add_ann
        
        
        self.table.clear()
        font = QtGui.QFont()
        font.setFamily(u"DejaVu Sans")
        font.setPointSize(5)
        self.table.horizontalHeader().setFont(font)
        self.table.verticalHeader().setFont(font)
        
        self.table.setColumnCount(t_row) # 次元数(tableはcolだけど値はrow!)+flag
        self.table.setRowCount(t_col) # 時間軸(tableはrowだけど値はcol!)

        # print "t:", t_row, t_col
        #self.table.setRowCount(data.shape[0])
        #self.table.setColumnCount(ann_num)
          
    
        # 軸の値をSet
        #for i in range(len(times)):
        #    jItem = QtGui.QTableWidgetItem(str(i))
        #    self.table.setHorizontalHeaderItem(i, jItem)
            
        
        hor = True
        for i in range(t_col):
            iItem = QtGui.QTableWidgetItem(str(i))
            self.table.setVerticalHeaderItem(i, iItem)
            self.table.verticalHeaderItem(i).setToolTip(str(i))
            #時間軸にデータを入れるなら↓
            #self.table.verticalHeaderItem(i).setToolTip(str(times[i]))
            
            for j in range(t_row):
                if hor == True:
                    jItem = QtGui.QTableWidgetItem(str(j))
                    self.table.setHorizontalHeaderItem(j, jItem)
                    self.table.horizontalHeaderItem(j).setToolTip(str(times[j]))
                    #print "%.10f"%times[j]
                    

                
                if i < d_col: #data(speak)の可視化
                    # 音声Dataがrmsの場合
                    # set_data:範囲は 0-1
                    set_data = data[j][i] #self.speakDecibelNorm(data[j][i])

                    #ON/OFF むちゃくちゃすぎる
                    #set_data = 1 if set_data > 0.3 else 0
                    #self.edited_speaks[j][i] = set_data
                    #一時的なOffset
                    #set_data = data_proc2.speakNorm(set_data, upper=0.75, lower=0.25)
                    #self.edited_speaks[j][i] = set_data
                    
                    #color_dataの範囲を0-255に変更
                    #color_data=int(set_data*255)
                    color_data = set_data
                    color_data = 1 if color_data > 0.5 else 0
                    color_data=int(color_data*255)
                    #print color_data
                    #print "at",color_data
                    color = [255-color_data]*3                   
                elif i >= d_col and i < t_col-add_flag: # annotationの可視化
                    #print i
                    """
                    set_data = anno[j][i-d_col] #anno(1000, 2)
                    if set_data == 0:
                        color = [255, 255, 255]
                    else:
                        color = [0, 0, 0]
                    """
                    set_data = anno[j][i-d_col] #anno(1000, 2)
                    set_data = 0 if set_data < 0 else set_data
                    set_data = 1 if set_data > 1 else set_data
                    color_data=int(set_data*255)
                    color = [255-color_data]*3
                    
                else: #flag
                    set_data = self.edited_flags[j][0]
                    if set_data == 0:
                        color = [255, 255, 255]
                    else:
                        color = [0, 0, 0]
                    
                self.table.setItem(i, j, QtGui.QTableWidgetItem())
                self.table.item(i, j).setBackground(QtGui.QColor(color[0],color[1],color[2]))
                self.table.item(i, j).setToolTip(str(set_data))
            hor = False
                          
        self.table.setVisible(False)
        self.table.resizeRowsToContents()
        self.table.resizeColumnsToContents()
        self.table.setVisible(True)


    # TableがClickされたとき
    def clickUpdateTable(self, cItem):

        self.tip = float(cItem.toolTip())
        self.r = cItem.row()
        self.c = cItem.column()
        print "r:",self.r,", c:",self.c, ". tip:",self.tip

        set_data = 0 if self.tip == 1 else 1 
        
        speak_dim = self.edited_speaks.shape[1]
        anno_dim = self.edited_annos.shape[1]
        
        if self.r < speak_dim:
            self.edited_speaks[self.c][self.r] = set_data
        elif self.r >= speak_dim and self.r < speak_dim+anno_dim:
            self.edited_annos[self.c][self.r-speak_dim] = set_data
        else:
            self.edited_flags[self.c][0] = set_data
        #indexes = self.table.selectedIndexes()
        #print indexes
        
        color = [[255, 255, 255], [0, 0, 0]]
                
        self.table.setItem(self.r, self.c, QtGui.QTableWidgetItem())
        self.table.item(self.r, self.c).setBackground(QtGui.QColor(color[set_data][0],color[set_data][1],color[set_data][2]))
        self.table.item(self.r, self.c).setToolTip(str(set_data))

        

        
        #jointsの可視化
        # self.vizJoint(self.c)
    def checkRangeData(self):
        set_row = int(self.checkUser.text()) #0~4 User1=2, User2=3
        start = int(self.checkSt.text())
        end = int(self.checkEd.text())

        color = [[255, 255, 255], [0, 0, 0]]
        table_offset = self.edited_speaks.shape[1]
        
        #変更するUserの指定
        user = set_row - table_offset
        #print user,  table_offset
        if user < 0:
            print "Not Change!"
            return

        print "Change [User:", user, ", Start:",start,", end:",end,"]"
        
        for i in range(start, end):
            
            get_data = self.edited_annos[i][user]  
        
            set_data = 0 if get_data == 1 else 1 
                
            self.table.setItem(set_row, i, QtGui.QTableWidgetItem())
            self.table.item(set_row, i).setBackground(QtGui.QColor(color[set_data][0],color[set_data][1],color[set_data][2]))
            self.table.item(set_row, i).setToolTip(str(set_data))

            self.edited_annos[i][user] = set_data
        
        
        
    def sliderChange(self, timeline):

        if len(self.input_joints)==0:
            print "now no data:", timeline
            return

        #self.table.selectColum(timeline)
        #self.table.verticalScrollBar().setValue(timeline)
        self.table.setCurrentCell(self.r, timeline)
        
        self.pubViz(timeline)

    """
    def activatedUpdateTable(self, cItem):
        row = cItem.row()
        col = cItem.column()
        print "row:", row,", col:", col#, ". tip:",self.tip        
    """
    
    # 生Dataの入力
    def inputData(self):
        filesize = int(self.frmSizeBox.text())
        print "Input raw data:", filesize
        self.fname = [str(self.txtSepFile.text())]
        input_data = data_proc2.load_persons(self.fname, annobool=False, datalen=filesize)
        #print input_data["times"][0] 
        # もし手をくわえるなら
        #input_data[2] =  data_proc.proc_anno(input_data[0], input_data[2], use_vote=True, use_speak=False)
        self.loadInputData(input_data)


    # 加工済のDataの入力  
    def inputProcedData(self):
        filesize = int(self.frmSizeBox.text())
        print "Input proced data", filesize
        self.fname = [str(self.txtSepFile.text())]
        input_data = data_proc2.load_proced_data_flag(self.fname, datalen=filesize)
        #print input_data["times"] 
        self.loadInputData(input_data)
        

    def loadInputData(self, input_data):
        self.input_joints = input_data["joints"]
        self.input_speaks = input_data["speaks"]
        datalen=self.input_speaks.shape[0]

        if input_data.has_key("annos"):
            self.input_annos = input_data["annos"]

            #とりあえずuser1だけ編集（0124_personsはまた別）
            #self.input_annos[:,:1] = np.zeros((self.input_annos.shape[0],1))
            """
            user_joint = self.input_joints[:,:36]
            user_anno = self.input_annos[:,:1]
            calc_user_anno =  data_proc2.proc_anno(user_joint, user_anno, use_vote=True, use_anno=False, threshold=-0.2)
            self.input_annos[:,:1] = calc_user_anno
            
            
            print "now_persons mode"
            user_joint = self.input_joints[:,36:]
            user_anno = self.input_annos[:,1:]
            calc_user_anno =  data_proc2.proc_anno(user_joint, user_anno, use_vote=True, use_anno=False, threshold=-0.2)            
            self.input_annos[:,1:] = calc_user_anno
            """
            
            #print "now model mode"
            self.input_annos[:,:] = np.round(self.input_annos[:,:])


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

        """
        diff_times = []
        for t in range(len(self.input_times)):
            if t == 0:
                diff_times.append(0)
            else:
                diff_times.append(self.input_times[t]-self.input_times[t-1])
                
        fps = np.round(1/float(np.mean(diff_times)))
        
        plt.plot(self.input_annos[:,0], label="person", color="red")
        plt.plot(self.input_annos[:,1], label="robot", color="green")
        plt.xlabel("Frame (fps:"+str(fps)+")")
        plt.title("speaker/listener result")
        plt.xlim(0,len(self.input_annos))
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.show()
        """ 
            
        print "joints shape:", self.input_joints.shape
        print "speaks shape:", self.input_speaks.shape
        print "annos shape:", self.input_annos.shape
        print "flags shape:", self.input_flags.shape
        print "times shape:", self.input_times.shape

        
        #for j in range(len(self.input_times)):
        #    print j,"%.10f"%self.input_times[j]
        self.timediff(self.input_times)
        
        self.edited_joints = copy.deepcopy(self.input_joints)#np.arrayでCopyされる
        self.edited_speaks = copy.deepcopy(self.input_speaks)#speakは編集しない(今のところ)
        self.edited_annos = copy.deepcopy(self.input_annos)
        self.edited_flags = copy.deepcopy(self.input_flags)
        self.edited_times = copy.deepcopy(self.input_times)
        
        
        #self.updateTable(self.input_joints)
        self.updateTable(self.edited_speaks, self.edited_annos, self.edited_times)
        #Sliderの最大値をset
        self.sld.setMaximum(datalen - 1)
        print "end"
        
    def outputData(self):
        
        name_json = str(self.txtOutputFile.text())
        keys = ["joints", "speaks", "annos", "flags", "times"]
        data = [self.edited_joints, self.edited_speaks, self.edited_annos, self.edited_flags, self.edited_times]

        """
        if len(keys) != len(data):
            print "Save false! keys len:"
        """
        
        data_proc2.save_data(name_json, keys, data)


    def timediff(self, times):

        fpss = []
        for i in range(len(times)-1):
            fps = 1/(times[i+1]-times[i])
            fpss.append(fps)
            #print i, fps
        fpss = np.array(fpss)
        #print "mean:",np.mean(fpss),",std:",np.std(fpss)
        
        
    def cutRangeData(self):

        start = int(self.cutStart.text())
        end = int(self.cutEnd.text())

        print "cut data:",start,"-",end

        self.edited_joints = self.edited_joints[start:end]
        self.edited_speaks = self.edited_speaks[start:end]
        self.edited_annos = self.edited_annos[start:end]
        self.edited_flags = self.edited_flags[start:end]
        self.edited_times = self.edited_times[start:end]
        
        print "joints shape:", self.edited_joints.shape
        print "speaks shape:", self.edited_speaks.shape
        print "annos shape:", self.edited_annos.shape
        print "flags shape:", self.edited_flags.shape
        print "times shape:", self.edited_times.shape
        
        self.updateTable(self.edited_speaks, self.edited_annos, self.edited_times)
        #Sliderの最大値をset
        self.sld.setMaximum(self.edited_joints.shape[0]-1)
        print "end"
        

    def directJoints(self):

        print "direct joints"
        
        size, dim = self.edited_joints.shape

        user1_nose_y_idx = 35+33+2

        pair = 1
        pair_stack = 0
        offset_y = 0.03
        
        for i in range(size):
            anno = self.edited_annos[i]
            
            if  anno[1] == 0:#聞き手なら

                if anno[0] == 1 and anno[2] == 0:
                    #print i,":user0",self.input_joints[i][user1_nose_y_idx]
                    self.edited_joints[i][user1_nose_y_idx] += offset_y
                    pair = 0
                    pair_stack = 0
                    #print self.input_joints[i][user1_nose_y_idx]
                elif anno[0] == 0 and anno[2] == 1:
                    #print i,":user2",self.input_joints[i][user1_nose_y_idx]
                    self.edited_joints[i][user1_nose_y_idx] -= offset_y
                    pair = 2
                    pair_stack = 0
                    #print self.input_joints[i][user1_nose_y_idx]
                else: #誰も話していないなら
                    pair_stack += 1

                    

            else:#話し手ならどこを向くか
                
                if pair == 0:
                    self.edited_joints[i][user1_nose_y_idx] += offset_y
                elif pair == 2:
                    self.edited_joints[i][user1_nose_y_idx] -= offset_y
                    
                
                
            
    # 引っ繰り返して追加
    def reverseData(self):

        #y方向だけ引っくり返す
        def reverse_y(joints):
            #(10000, 36)
            rev = np.array([[1,-1,1]*12]*joints.shape[0])
            return joints*rev
        
        print "reverseData before:",self.edited_joints.shape, self.edited_annos.shape
        datalen = self.edited_annos.shape[0]
        
        if self.anno_dim == 2:
            j_r = np.hstack((self.edited_joints[:,36:], self.edited_joints[:,:36]))
            self.edited_joints = np.vstack((self.edited_joints, j_r))

            s_r = np.hstack((self.edited_speaks[:,1].reshape(datalen,1), self.edited_speaks[:,0].reshape(datalen,1)))
            self.edited_speaks = np.vstack((self.edited_speaks, s_r))
            
            a_r = np.hstack((self.edited_annos[:,1].reshape(datalen,1), self.edited_annos[:,0].reshape(datalen,1)))
            self.edited_annos = np.vstack((self.edited_annos, a_r))

            self.edited_flags = np.vstack((self.edited_flags, self.edited_flags))
            self.edited_times = np.append(self.edited_times, self.edited_times)            
        else:
            print "bad shape"

        #print self.edited_times.shape
        self.updateTable(self.edited_speaks, self.edited_annos,  self.edited_times)
        #Sliderの最大値をset
        self.sld.setMaximum(self.edited_joints.shape[0]-1)

        print "reverseData after:",self.edited_joints.shape,self.edited_speaks.shape,self.edited_annos.shape 

        


        
    def rviz_obj(self, obj_id, obj_ns, obj_type, obj_size, obj_color=[0, 0, 0, 0], obj_life=0):
        obj = Marker()
        obj.header.frame_id, obj.header.stamp = "camera_link", rospy.Time.now()
        obj.ns, obj.action, obj.type = str(obj_ns), 0, obj_type
        obj.scale.x, obj.scale.y, obj.scale.z = obj_size[0], obj_size[1], obj_size[2]
        obj.color = obj_color
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

    
    def set_vizmsg_point(self, u, data,  color, psize, ofs, addx=0, addy=0, rotate=False):
        pmsg = self.rviz_obj(u, 'p'+str(u), 7, [psize, psize, psize], color, 0)
        points = []
        for p in range(data.shape[1]/ofs):
            points.append(self.set_point([data[0, p*ofs],
                                          data[0, p*ofs+1],
                                          data[0, p*ofs+2]],
                                         addx=addx, addy=addy, rotate=rotate))
        pmsg.points = points
        return pmsg
    

    def set_vizmsg_line(self, u, data,  color, lsize, llist, addx=0, addy=0, rotate=False):
        lmsg = self.rviz_obj(u, 'l'+str(u), 5, [lsize, lsize, lsize], color, 0)  
        for ls in llist:
            for l in range(len(ls)-1):
                for add in range(2):
                    #print person[0, ls[l+add]], ls[l+add], l, add
                    linepoint=self.set_point([data[0,ls[l+add]*3],
                                              data[0,ls[l+add]*3+1],
                                              data[0,ls[l+add]*3+2]],
                                             addx=addx, addy=addy, rotate=rotate)
                    lmsg.points.append(linepoint)
        return lmsg
    
        
        
    def pubViz(self, tl):
        #print "pub Viz:", tl
        # drawing
        msgs = MarkerArray()
        amsg = Float64MultiArray()
        
        per_js = []
        dim_p = 36
        dim = len(self.edited_joints[tl])
        for i in range(dim/dim_p):
            per_js.append(self.edited_joints[tl,dim_p*i:dim_p*(i+1)].reshape(1, dim_p))
            
        ofs = 3

        #ofs_xyr = [[-0.5, -0.5, 1], [0.5, 0, 0], [-0.5, 0.5, 1]]
        ofs_xyr = [[-1, 0, 1], [0, 0, 0]]
        for u, (person, speak, anno) in enumerate(zip(per_js, self.edited_speaks[tl], self.edited_annos[tl])):
            # ---Person points---
            offset = 0
            psize = speak*0.05
            #psize = 0.03
            #psize = 0.05 if anno > 0.7 else 0.03
            pmsg = self.set_vizmsg_point(u, person, self.carray[0], psize, ofs,
                                         addx=ofs_xyr[u][0], addy=ofs_xyr[u][1], rotate=ofs_xyr[u][2])
            msgs.markers.append(pmsg)
            
            # ---Person lines---
            lsize = 0.03 if anno > 0.5 else 0.01
            cid = 3 if anno > 0.5 else 2
            lmsg = self.set_vizmsg_line(u, person, self.carray[cid], lsize, self.llist,
                                        addx=ofs_xyr[u][0], addy=ofs_xyr[u][1], rotate=ofs_xyr[u][2])
            msgs.markers.append(lmsg)
            
            # ------text------
            tidx, tsize = 3, 0.1 # tidx=3 is Head
            tmsg = self.rviz_obj(u, 't'+str(u), 9, [tsize, tsize, tsize], self.carray[u], 0)
            tmsg.pose.position = self.set_point([person[0, tidx*ofs], person[0, tidx*ofs+1], person[0, tidx*ofs+2]],
                                                addx=ofs_xyr[u][0], addy=ofs_xyr[u][1], addz=0.3)
            tmsg.pose.orientation.w = 1
            tmsg.text = "User"+str(u)+":"+str(speak)
            msgs.markers.append(tmsg)

            amsg.data.append(speak)

            
        # ------text------
        if self.edited_times[tl] > 0.00001:
            tidx, tsize = 3, 0.1 # tidx=3 is Head
            tmsg = self.rviz_obj(u, 'time'+str(u), 9, [tsize, tsize, tsize], self.carray[u], 0)
            tmsg.pose.position = self.set_point([-0.5, 0, 0])
            tmsg.pose.orientation.w = 1
            tmsg.text = "time: "+str(self.edited_times[tl])
            msgs.markers.append(tmsg)
        
        
        self.mpub.publish(msgs)

        
        #print  pred[1].data[0, 0]

        self.speakpub.publish(amsg)

        
        
def main():
    app = QtGui.QApplication(sys.argv)
    anotation = ANNOTATION()
    #graph = GRAPH()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
