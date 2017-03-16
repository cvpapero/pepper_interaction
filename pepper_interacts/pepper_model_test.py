# coding: utf-8


import argparse
import copy
from datetime import datetime 
import glob
import time
import sys
import json
import numpy as np
import six
import os
import tqdm

import net
import data_proc2
import rviz_box

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers


import rospy
from humans_msgs.msg import Humans
from humans_msgs.msg import AudioData
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', '-trd', default='3seqx2_raw-spk-40-60',
                    help='filename')
parser.add_argument('--pred_dir', '-p', default="LSTM_ANN_20170209_223520/",
                    help='filename')
parser.add_argument('--use_model', '-m', default="final",
                    help='use model epXXX or final')
args = parser.parse_args()

train_file = glob.glob(args.train_dir+"/*")
train_data = data_proc2.load_proced_data(train_file) #(joints, speaks, annos)
train_joints, train_speaks, train_annos = train_data[0], train_data[1], train_data[2]

st = 0
train_joints, train_speaks, train_annos = train_joints[st:], train_speaks[st:], train_annos[st:] 
rospy.init_node("connect_pepper", anonymous=True)
mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

#dim_j = 72
#dim_v = 24
#dim_s = 2
#dim_a = 2

folder_name = args.pred_dir
param_data = json.load(open(folder_name+"/param.json"))
print "version:", param_data["version"]
dim_j = int(param_data["net"]["dim_j"])
dim_v = int(param_data["net"]["dim_v"])
dim_s = int(param_data["net"]["dim_s"])
dim_a = int(param_data["net"]["dim_a"])
dim_h = int(param_data["net"]["dim_h"])
model_name = folder_name+"/"+param_data["net"]["name"]+"_"+args.use_model+".model"

model = net.LSTM_ANN(dim_j+dim_v+dim_s, dim_a, dim_h, train=False)
serializers.load_npz(model_name, model)
        
offset = 3

pre_person_joint = np.random.rand(1, dim_j/2)
pre_robot_joint = np.random.rand(1, dim_j/2)

pre_person_speak = 0
pre_robot_speak = 0

pre_person_anno = 0
pre_robot_anno = 0


tl = 0
past_joints_data =  train_joints[0, 0:dim_j].reshape(1, dim_j).astype(np.float32)

inf_mode = 0

while not rospy.is_shutdown():
    print tl
    now_time = rospy.get_time()
    #fps = 1/(now_time - self.past_time)
    #self.past_time = now_time
    #rospy.loginfo("now recog human:%s, fps:%s",str(len(msg.human)), str(fps))
    
    msgs = MarkerArray()
    ofs = 3
        
    

    if inf_mode == 0:
        #annotation推定
        person_joint = train_joints[tl, 0:dim_j/2].reshape(1, dim_j/2).astype(np.float32)
        robot_joint = train_joints[tl, dim_j/2:].reshape(1, dim_j/2).astype(np.float32)
        person_speak = train_speaks[tl, 0]
        robot_speak = train_speaks[tl, 1]
        person_anno = pre_person_anno 
        robot_anno = pre_robot_anno
        caption = "INFER:speaker"
    elif inf_mode == 1:
        #position, speak推定
        person_joint = pre_person_joint.reshape(1, dim_j/2).astype(np.float32)#(1, 36)
        robot_joint = pre_robot_joint.reshape(1, dim_j/2).astype(np.float32)#(1, 36)
        person_speak = pre_person_speak
        robot_speak = pre_robot_speak
        person_anno = train_annos[tl, 0]
        robot_anno = train_annos[tl, 1]
        caption = "INFER:pos&voice"
    else:
        # the other推定
        """
        person_joint = train_joints[tl, 0:dim_j/2].reshape(1, dim_j/2).astype(np.float32)
        robot_joint = pre_robot_joint.reshape(1, dim_j/2).astype(np.float32)#(1, 36)
        person_speak = train_speaks[tl, 0]
        robot_speak = pre_robot_speak
        person_anno = pre_person_anno 
        robot_anno = pre_robot_anno
        caption = "INFER:person2(pos,voice)&speaker"
        """
        person_joint = pre_person_joint.reshape(1, dim_j/2).astype(np.float32)#(1, 36)
        robot_joint = train_joints[tl, dim_j/2:].reshape(1, dim_j/2).astype(np.float32)
        person_speak = pre_person_speak 
        robot_speak = train_speaks[tl, 1]
        person_anno = pre_person_anno 
        robot_anno = pre_robot_anno
        caption = "INFER:person1(pos,voice)&speaker"
        
    # Set Joints        
    joints_data = np.hstack((person_joint, robot_joint)) #(1, 72)
    
    # Set Velocity
    now_joints_data = joints_data.reshape(dim_j/offset, offset)
    past_joints_data = past_joints_data.reshape(dim_j/offset, offset)
    vels_data = data_proc2.calc_velocity_online(now_joints_data, past_joints_data).reshape(1, dim_v)
    
    # Set Speaks
    speaks_data = np.array([[person_speak, robot_speak]]).astype(np.float32)
             
    # Set Annotations
    annos_data = np.array([[person_anno, robot_anno]]).astype(np.float32)

    
    # Drawing
    rbox = rviz_box.RvizBox()
    viz_joints = [person_joint, robot_joint]
    viz_speaks = [pre_person_speak, pre_robot_speak]
    viz_annos = [person_anno, robot_anno]
    offsets = [0, -1] # defaultだとだぶるからずらす
    rotates = [False, True] # robotを回転させる
    voice_text = ["inf_voice", "inf_voice"]
    speaker_text = ["inf_speaker", "inf_speaker"]
    texts = ["Person 1", "Person 2"]
    
    for u in range(len(viz_joints)):
        
        # ---Points---
        psize = viz_speaks[u]*0.05
        pmsg = rbox.set_vizmsg_point(u, viz_joints[u], rbox.carray[0], psize, ofs,
                                             addx=offsets[u], rotate=rotates[u])
        msgs.markers.append(pmsg)
        # ---Lines---
        lsize = 0.01
        c_id = 3 if viz_annos[u] > 0.15 else 2
        lmsg = rbox.set_vizmsg_line(u, viz_joints[u], rbox.carray[c_id], lsize, rbox.llist,
        addx=offsets[u], rotate=rotates[u])
        msgs.markers.append(lmsg)
        #---Name Text---
        tidx, tsize = 3, 0.1 # tidx=3 is Head
        tmsg = rbox.rviz_obj(u, 't'+str(u), 9, [tsize, tsize, tsize], rbox.carray[c_id], 0)
        tmsg.pose.position = rbox.set_point([viz_joints[u][0, tidx*ofs],
                                             viz_joints[u][0, tidx*ofs+1],
                                             viz_joints[u][0, tidx*ofs+2]],
                                            addz=0.3, addx=offsets[u], rotate=rotates[u])
        tmsg.pose.orientation.w = 1
        tmsg.text = texts[u]#+":"+str(round(viz_speaks[u]))
        msgs.markers.append(tmsg)
        
        #---voice---
        tidx, tsize = 3, 0.1 # tidx=3 is Head
        c_id = 3 if viz_annos[u] > 0.5 else 2
        
        tmsg = rbox.rviz_obj(u, 'vt'+str(u), 9, [tsize, tsize, tsize], rbox.carray[c_id], 0)
        tmsg.pose.position = rbox.set_point([viz_joints[u][0, tidx*ofs],
                                             viz_joints[u][0, tidx*ofs+1],
                                             viz_joints[u][0, tidx*ofs+2]],
                                            addz=0.5, addx=offsets[u], rotate=rotates[u])
        tmsg.pose.orientation.w = 1
        tmsg.text = voice_text[u]+":"+str(round(viz_speaks[u],4))
        msgs.markers.append(tmsg)
        
        #---speaker---
        tmsg = rbox.rviz_obj(u, 'st'+str(u), 9, [tsize, tsize, tsize], rbox.carray[c_id], 0)
        tmsg.pose.position = rbox.set_point([viz_joints[u][0, tidx*ofs],
                                             viz_joints[u][0, tidx*ofs+1],
                                             viz_joints[u][0, tidx*ofs+2]],
                                            addz=0.6, addx=offsets[u], rotate=rotates[u])
        tmsg.pose.orientation.w = 1
        tmsg.text = speaker_text[u]+":"+str(round(viz_annos[u],4))
        msgs.markers.append(tmsg)

        """
        #---mode 2---
        if inf_mode == 2:
            if u == 0:
                tsize = 0.1
                tmsg = rbox.rviz_obj(u, 'mode'+str(u), 9, [tsize, tsize, tsize], rbox.carray[1], 0)
                tmsg.pose.position = rbox.set_point([0.5, 0, 0])
                tmsg.pose.orientation.w = 1
                tmsg.text = "act_voice: "
                msgs.markers.append(tmsg)
        """

    #---title---
    tsize = 0.08
    tmsg = rbox.rviz_obj(u, 'tl'+str(u), 9, [tsize, tsize, tsize], rbox.carray[1], 0)
    tmsg.pose.position = rbox.set_point([-0.5, 0, -0.7])
    tmsg.pose.orientation.w = 1
    tmsg.text = caption
    msgs.markers.append(tmsg)
    
    mpub.publish(msgs)         
            
    
    # 速度計算用に保存
    past_joints_data = joints_data
            
    # Prediction
    #print joints_data.shape, vels_data.shape, speaks_data.shape
    jvs_data = np.hstack((joints_data, vels_data, speaks_data))            
    pred = model(jvs_data, annos_data, 0, 0)
    pre_person_joint = pred[0].data[0, 0:dim_j/2] #(1, 36)
    pre_robot_joint = pred[0].data[0, dim_j/2:dim_j] #(1, 36)
    pre_person_speak = pred[0].data[0, dim_j+dim_v:dim_j+dim_v+dim_s/2][0]
    pre_robot_speak = pred[0].data[0, dim_j+dim_v+dim_s/2:]
    pre_robot_anno = pred[1].data[0, 1]            
    pre_person_anno = pred[1].data[0, 0]

    tl += 1
    time.sleep(0.03)
