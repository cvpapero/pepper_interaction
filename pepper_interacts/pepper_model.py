#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2017/1/22
pepper_speakと合体

2017/1/17
rvizの部分を大体package化した

2017/1/16
学習したModelを用いてPepperを動かす
発話と話者AnnotationをPub
"""


import argparse
import numpy as np
import json
import os
import copy
import sys
import threading
import time

import chainer
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
from geometry_msgs.msg import Point

from std_srvs.srv import Empty
from std_srvs.srv import EmptyResponse

from naoqi import ALProxy

import net
import data_proc2
import calc_angles
import rviz_box
import text_example


#はいぱらはおはじだけどそうていどおりにうごくもでる
#LSTM_ANN_20170209_223520/


parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', '-p', default="LSTM_ANN_20170113_195638",
                    help='filename')
parser.add_argument('--use_model', '-m', default="final",
                    help='use model epXXX or final')
parser.add_argument('--save_name', '-n', default="",
                    help='save file name')
parser.add_argument('--topic', '-t', default="/humans/kinect_v2", 
                    help='topic')
args = parser.parse_args()


class PepperInteract():
    def __init__(self):

        self.topic = args.topic
        print "topic:", args.topic
        self.ksub = rospy.Subscriber(self.topic, Humans, self.callback)
        self.ssub = rospy.Subscriber('/humans/audio/1', AudioData, self.speakCb)
        self.apub = rospy.Publisher('/person_robot/anno', Float64MultiArray, queue_size=10)
        self.mpub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

        # No Speak Service
        no_speak_srv = rospy.Service('/robots/anno', Empty, self.no_speak_srv)
        #no_speak_srv = rospy.Service('/robots/anno', SetBool, self.no_speak_srv)
        self.past_time = rospy.get_time()
        
        self.stop_pepper()
        self.init_pepper_param()
        self.set_network()

        # use joints ids
        self.sidx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20, 25]  

        self.robot_end_srv = False

        self.save_joints, self.save_speaks, self.save_annos, self.save_times = [], [], [], []

        #発話の制御
        self.robot_speaker_state = False
        self.th = 0.5
        self.texts = text_example.input_texts()
        
        threading.Thread(target=self.speaker_controller).start()
        threading.Thread(target=self.speaking_controller).start()
        
    def init_pepper_param(self):

        # Speaker
        self.speak = ALProxy("ALTextToSpeech", "133.19.23.152", 9559)
        self.speak.setLanguage("Japanese")
        self.audiodevice = ALProxy("ALAudioDevice", "133.19.23.152", 9559)
        

        # Joint Movement
        self.motion = ALProxy("ALMotion","133.19.23.152",9559)
     
        head = ["HeadPitch", "HeadYaw"] 
        arm_L = ["LShoulderRoll", "LShoulderPitch", "LElbowRoll", "LElbowYaw", "LWristYaw"]
        arm_R = ["RShoulderRoll", "RShoulderPitch", "RElbowRoll", "RElbowYaw", "RWristYaw"]
        lower = ["HipRoll", "HipPitch", "KneePitch"]
        hands = ["LHand", "RHand"]
        
        self.angle_names = []
        self.angle_names.extend(head)
        self.angle_names.extend(arm_L)
        self.angle_names.extend(arm_R)
        self.angle_names.extend(lower)
        self.angle_names.extend(hands)
        
        #なめらかに
        stack_len=10
        self.heps = [0 for i in range(stack_len)]
        self.heys = [0 for i in range(stack_len)]
        self.lsrs = [0 for i in range(stack_len)]
        self.lsps = [0 for i in range(stack_len)]
        self.lers = [0 for i in range(stack_len)]
        self.leys = [0 for i in range(stack_len)]
        self.lwys = [0 for i in range(stack_len)]
        self.rsrs = [0 for i in range(stack_len)]
        self.rsps = [0 for i in range(stack_len)]
        self.rers = [0 for i in range(stack_len)]
        self.reys = [0 for i in range(stack_len)]
        self.rwys = [0 for i in range(stack_len)]
        self.hirs = [0 for i in range(stack_len)]
        self.hips = [0 for i in range(stack_len)]
        self.knps = [0 for i in range(stack_len)]
        
        
    def stop_pepper(self):
        ba = ALProxy("ALBasicAwareness","133.19.23.152",9559)
        ba.stopAwareness()
        am = ALProxy("ALAutonomousMoves","133.19.23.152",9559)
        am.setBackgroundStrategy("none");
        am.setExpressiveListeningEnabled(0);


    def set_network(self):
        folder_name = args.pred_dir
        param_data = json.load(open(folder_name+"/param.json"))
        print "version:", param_data["version"]
        self.dim_j = int(param_data["net"]["dim_j"])
        self.dim_v = int(param_data["net"]["dim_v"])
        self.dim_s = int(param_data["net"]["dim_s"])
        self.dim_a = int(param_data["net"]["dim_a"])
        self.dim_h = int(param_data["net"]["dim_h"])
        model_name = folder_name+"/"+param_data["net"]["name"]+"_"+args.use_model+".model"
        print "Set Model:", model_name
        self.model = net.LSTM_ANN(self.dim_j+self.dim_v+self.dim_s, self.dim_a, self.dim_h, train=False)
        serializers.load_npz(model_name, self.model)

        # init robot data
        self.pre_robot_joint = np.random.rand(1, self.dim_j/2) 
        self.pre_robot_speak = 0
        self.pre_robot_anno = 0
        # for velocity
        self.offset = 3
        self.past_joints_data = np.zeros(self.dim_j)
        #
        self.pre_person_speak = 0
        self.pre_person_anno = 0


    def set_angles(self, rbt):

        def ave_angle(data):
            return sum(data)/float(len(data))
        
        caan = calc_angles.CalcAngles()
        hep, hey, lsr, lsp, ler, ley, lwy, rsr, rsp, rer, rey, rwy, hir, hip, knp =  caan.set_angles_pose(rbt)
        
        self.heps.append(hep)
        self.heps.pop(0)
        hep = ave_angle(self.heps)

        self.lsps.append(lsp)
        self.lsps.pop(0)
        lsp = ave_angle(self.lsps)

        self.rsps.append(rsp)
        self.rsps.pop(0)
        rsp = ave_angle(self.rsps)

        self.lsrs.append(lsr)
        self.lsrs.pop(0)
        lsr = ave_angle(self.lsrs)

        self.rsrs.append(rsr)
        self.rsrs.pop(0)
        rsr = ave_angle(self.rsrs)

        self.lers.append(ler)
        self.lers.pop(0)
        ler = ave_angle(self.lers)

        self.rers.append(rer)
        self.rers.pop(0)
        rer = ave_angle(self.rers)

        self.hirs.append(hir)
        self.hirs.pop(0)
        hir = ave_angle(self.hirs)

        self.hips.append(hip)
        self.hips.pop(0)
        hip = ave_angle(self.hips)

        rs = [hep, hey, lsr, lsp, ler, ley, lwy, rsr, rsp, rer, rey, rwy, hir, hip, knp, 0, 0]
        return rs


    #annotationに基く話し手聞き手の制御
    def speaker_controller(self):
        while not rospy.is_shutdown():
            if self.pre_robot_anno <= self.th:
                # 聞き手ならvolume調節
                vol = int(round(self.pre_robot_speak*40))
                self.audiodevice.setOutputVolume(vol)
            else:
                # 話し手ならvolumeの下限を固定
                vol = int(round(self.pre_robot_speak*40))
                vol = vol if vol > 30 else 30
                #vol = 45
                self.audiodevice.setOutputVolume(vol)
            
            if self.pre_person_anno > 0.2 and self.robot_speaker_state == True:
                #robotが話し手であり、話の途中にPersonが話し手になったら
                self.robot_speaker_state = False
                self.speak.stopAll()
            time.sleep(0.01)
                
    def speaking_controller(self):
        while not rospy.is_shutdown():
            if self.pre_robot_anno < self.th and self.pre_robot_speak > self.th:
                # 聞き手の場合
                self.robot_speaker_state = False
                hai = "はい"
                self.speak.say(hai)
                
            elif self.pre_robot_anno > self.th: #and self.pre_robot_speak > self.th:
                # 話し手の場合
                # say()は一度始まったら読み上げは終らない。別Threadで割込み終了させる
                self.robot_speaker_state = True
                text_idx = np.random.randint(len(self.texts))
                text = self.texts[text_idx]#らんだむに選ばれた文章を読上げ
                self.speak.say(text)
                time.sleep(3)
            time.sleep(0.01)
            
    # pepper_speakがfalseになったら(pepperが話をおえたら)
    def no_speak_srv(self, req):
        self.robot_end_srv = True
        print "no_speak_srv"
        #self.model.reset_state()
        return EmptyResponse()
    
    def set_position(self, joints, sidx, dim_x):
        pose = []
        for pos in joints: 
            pose.append(pos.position.x)
            pose.append(pos.position.y)
            pose.append(pos.position.z)
        pose = data_proc2.select_data_online(np.array(pose, dtype=np.float32), sidx)
        return data_proc2.normalize_data(pose, 3)[0].reshape(1, dim_x/2).astype(np.float32)

    # HeadSetからの音声
    def speakCb(self, msg):
        human_speak = data_proc2.speakDecibelNorm(msg.data)
        #self.human_speak = human_speak
        self.human_speak = 1 if human_speak > self.th else 0


    def callback(self, msg):
        
        now_time = rospy.get_time()
        fps = 1/(now_time - self.past_time)
        self.past_time = now_time
        rospy.loginfo("now recog human:%s, fps:%s",str(len(msg.human)), str(fps))

        msgs = MarkerArray()
        ofs = 3
        
        for u, human in enumerate(msg.human):

            #Userのひとりめだけ処理
            if u > 0:
                continue
            
            # Set Joints
            person_joint = self.set_position(msg.human[0].body.joints, self.sidx, self.dim_j)
            robot_joint = self.pre_robot_joint.reshape(1, self.dim_j/2).astype(np.float32)#(1, 36)
            joints_data = np.hstack((person_joint, robot_joint)) #(1, 72)

            # Set Velocity
            now_joints_data = joints_data.reshape(self.dim_j/self.offset, self.offset)
            self.past_joints_data = self.past_joints_data.reshape(self.dim_j/self.offset, self.offset)
            vels_data = data_proc2.calc_velocity_online(now_joints_data, self.past_joints_data).reshape(1, self.dim_v)
            #vels_data = joints_data - self.past_joints_data
            
            # Set Speaks
            person_speak = self.human_speak
            #person_speak = msg.human[0].body.is_speaked #kinectの音声を使うなら
            robot_speak = self.pre_robot_speak    
            speaks_data = np.array([[person_speak, robot_speak]]).astype(np.float32)
             
            # Set Annotations
            person_anno = self.pre_person_anno
            robot_anno = self.pre_robot_anno
            annos_data = np.array([[person_anno, robot_anno]]).astype(np.float32)

            
            """
            if self.robot_end_srv == True:
                print "robot speak end"
                robot_speak = 0
                robot_anno = 0
                self.robot_end_srv = False
            """
            
            # Drawing
            rbox = rviz_box.RvizBox()
            viz_joints = [person_joint, robot_joint]
            viz_speaks = [self.pre_person_speak, self.pre_robot_speak]
            viz_annos = [person_anno, robot_anno]
            offsets = [0, -1] # defaultだとだぶるからずらす
            rotates = [False, True] # robotを回転させる
            voice_text = ["estimate_voice", "estimate_voice"]
            speaker_text = ["estimate_speaker", "estimate_speaker"]
            
            for u in range(len(viz_joints)):
                """
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
                """
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

                        
            self.mpub.publish(msgs)         
            
            
            # 関節角度に変換
            angle_values = self.set_angles(robot_joint[0])
            #print angle_values
            self.motion.setAngles(self.angle_names, angle_values, 0.2)

            # 速度計算用に保存
            self.past_joints_data = joints_data
            
            # Prediction
            jvs_data = np.hstack((joints_data, vels_data, speaks_data))            
            pred = self.model(jvs_data, annos_data, 0, 0)
            self.pre_robot_joint = pred[0].data[0, self.dim_j/2:self.dim_j] #(1, 36)
            self.pre_person_speak = pred[0].data[0, self.dim_j+self.dim_v:self.dim_j+self.dim_v+self.dim_s/2]
            self.pre_robot_speak = pred[0].data[0, self.dim_j+self.dim_v+self.dim_s/2:]
            self.pre_robot_anno = pred[1].data[0, 1]            
            self.pre_person_anno = pred[1].data[0, 0]

            
            # 予測結果をPub
            amsgs = Float64MultiArray()
            amsgs.data.append(self.pre_person_anno)
            amsgs.data.append(self.pre_robot_anno)
            amsgs.data.append(person_speak) #人の発話
            amsgs.data.append(self.pre_robot_speak)
            self.apub.publish(amsgs)
            
            
            # data save
            if len(args.save_name)>0:
                self.save_joints.append(joints_data[0].tolist())
                self.save_speaks.append(speaks_data[0].tolist())
                self.save_annos.append(annos_data[0].tolist())
                self.save_times.append(now_time)
                

def save(obj, name):
    dict = {
        "joints":obj.save_joints,
        "speaks":obj.save_speaks,
        "annos":obj.save_annos,
        "times":obj.save_times,
    }
    open(name, 'w').write(json.dumps(dict))
    print "save:", name
    print len(obj.save_joints), len(obj.save_speaks), len(obj.save_annos), len(obj.save_times)


def main():
    rospy.init_node("connect_pepper", anonymous=True)
    #ros_topic = rospy.get_param("~topic", "/humans/kinect_v2")  
    pi = PepperInteract()

    while not rospy.is_shutdown():
        rospy.spin()

    if len(args.save_name)>0:
        file_name = args.save_name+".json"
        save(pi, file_name)

    pi.speak.stopAll()
    
    print "shoutdown"
        
if __name__ == "__main__":
    print "start"
    main()
