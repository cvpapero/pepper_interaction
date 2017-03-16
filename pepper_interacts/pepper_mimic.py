#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2017/1/19
$ rosservice call /robots/anno "{}" 
これをすると話し手と聞き手が交換
さらに聞き手状態で首を0.25以上まげると「はい」という


2017/1/14
quaternion
https://github.com/malaybaku/KinectForPepper/blob/master/src/KinectForPepper/Models/BodyToJointAngle.cs

まだできてないこと
速度変更

"""


#import argparse
import numpy as np
import json
import os
import copy
import sys
import math
import threading
import time

import rospy
from humans_msgs.msg import Humans
from humans_msgs.msg import AudioData

from std_srvs.srv import Empty
from std_srvs.srv import EmptyResponse

from naoqi import ALProxy

import data_proc2
import calc_angles
import text_example

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pred_dir', '-p', default="20161108_195834_LSTM_ANN",
                    help='filename')
parser.add_argument('--use_model', '-m', default="final",
                    help='use model epXXX or final')
parser.add_argument('--save_name', '-n', default="",
                    help='save file name')
parser.add_argument('--topic', '-t', default="/humans/kinect_v2", 
                    help='topic')
args = parser.parse_args()


class ConnectPepper():
    def __init__(self):

        self.topic = args.topic
        print "topic:", self.topic
        self.ksub = rospy.Subscriber(self.topic, Humans, self.callback)
        #self.ssub = rospy.Subscriber('/humans/audio/1', AudioData, self.speakCb)
        
        self.stop_pepper()
        


        
        self.sidx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20, 25]  

        self.motion = ALProxy("ALMotion","133.19.23.152",9559)
        self.speak = ALProxy("ALTextToSpeech", "133.19.23.152", 9559)
        self.speak.setLanguage("Japanese")
        self.audiodevice = ALProxy("ALAudioDevice", "133.19.23.152", 9559)
        self.audiodevice.setOutputVolume(45)

        
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

        
        speak_srv = rospy.Service('/robots/anno', Empty, self.speak_srv)
        self.texts = text_example.input_texts()
        self.robot_speaker = False
        self.nod = False
        threading.Thread(target=self.speakCB).start()
        
        
    def stop_pepper(self):
        ba = ALProxy("ALBasicAwareness","133.19.23.152",9559)
        ba.stopAwareness()
        
        am = ALProxy("ALAutonomousMoves","133.19.23.152",9559)
        am.setBackgroundStrategy("none");
        am.setExpressiveListeningEnabled(0);


    def set_pose_and_ort(self, joints, sidx):
        
        def select_data_online(data, idx, ofs):
            sidx = [sid*ofs+i  for sid in idx for i in range(ofs)]
            return data[sidx]
        
        poses, orts = [], []
        for pos in joints:
            #print ort
            poses.append(pos.position.x)
            poses.append(pos.position.y)
            poses.append(pos.position.z)
            orts.append(pos.orientation.x)
            orts.append(pos.orientation.y)
            orts.append(pos.orientation.z)
            orts.append(pos.orientation.w)
        #print orit
        pose = select_data_online(np.array(poses), sidx, 3)
        pose = data_proc2.normalize_data(pose, 3)[0]
        ort = select_data_online(np.array(orts), sidx, 4)
        
        return pose, ort
    
    



    
    def set_angles(self, positions, orientations):
        caan = calc_angles.CalcAngles()
        #print body
        poses = positions.reshape(len(self.sidx), 3)
        orts = orientations.reshape(len(self.sidx), 4)
        
        shoulder_l, elbow_l, wrist_l = orts[4], orts[5], orts[6]
        lsp, lsr, ley, ler = caan.set_arm_angles(shoulder_l, elbow_l, wrist_l, "left")
        lsr = lsr - 15. * np.pi/180.

        shoulder_r, elbow_r, wrist_r = orts[7], orts[8], orts[9]               
        rsp, rsr, rey, rer = caan.set_arm_angles(shoulder_r, elbow_r, wrist_r, "right")
        rsr = rsr + 15. * np.pi/180.

        spine = orts[1]
        hip = caan.set_hip(spine)
        hir = 0

        nose_pos = poses[11]
        head_pos = poses[3]
        spine_pos = poses[10]

        #heyはNormalizeされたpositionならちゃんと動作する
        hep, hey = caan.set_head(nose_pos, head_pos, spine_pos)
        hey = 0
        
        lwy = caan.set_wrist_yaws(elbow_l, wrist_l, "left")    
        rwy = caan.set_wrist_yaws(elbow_r, wrist_r, "right")
       
        knp = 0

        
        self.heps.append(hep)
        self.heps.pop(0)
        hep = sum(self.heps)/float(len(self.heps))
        #print hep
        #頷くと大体0.25
        self.nod = True if hep > 0.25 else False
        
        self.heys.append(hey)
        self.heys.pop(0)
        hey = sum(self.heys)/float(len(self.heys))

        self.lsrs.append(lsr)
        self.lsrs.pop(0)
        lsr = sum(self.lsrs)/float(len(self.lsrs))
        
        self.lsps.append(lsp)
        self.lsps.pop(0)
        lsp = sum(self.lsps)/float(len(self.lsps))

        self.lers.append(ler)
        self.lers.pop(0)
        ler = sum(self.lers)/float(len(self.lers))
        
        self.leys.append(ley)
        self.leys.pop(0)
        ley = sum(self.leys)/float(len(self.leys))

        self.rsrs.append(rsr)
        self.rsrs.pop(0)
        rsr = sum(self.rsrs)/float(len(self.rsrs))
        
        self.rsps.append(rsp)
        self.rsps.pop(0)
        rsp = sum(self.rsps)/float(len(self.rsps))

        self.rers.append(rer)
        self.rers.pop(0)
        rer = sum(self.rers)/float(len(self.rers))
        
        self.reys.append(rey)
        self.reys.pop(0)
        rey = sum(self.reys)/float(len(self.reys))


        #left_hand_state
        #right_hand_state
        
        rs = [hep, hey, lsr, lsp, ler, ley, lwy, rsr, rsp, rer, rey, rwy, hir, hip, knp, 0, 0]

        return rs


    # pepper_speakがfalseになったら(pepperが話をおえたら)
    def speak_srv(self, req):
        self.robot_speaker = not self.robot_speaker
        print "robot_speaker:", self.robot_speaker

        if self.robot_speaker==False:
            self.speak.stopAll()
            
        return EmptyResponse()
    
    
    def speakCB(self):
        
        while not rospy.is_shutdown():
            #print "speak callback:", self.robot_speaker
            if self.robot_speaker:
                text_idx = np.random.randint(len(self.texts))
                text = self.texts[text_idx]#Randomに選ばれた文章を読上げ
                self.speak.say(text)
                self.robot_speaker=False
                #time.sleep(5)
            if self.robot_speaker==False and self.nod:
                text = "はい"
                self.speak.say(text)
    
    
  
    def callback(self, msg):
        
        rospy.loginfo("now recog human:%s",str(len(msg.human)))

        for u, human in enumerate(msg.human):
            pos, ort = self.set_pose_and_ort(msg.human[u].body.joints, self.sidx)
            angle_values = self.set_angles(pos, ort)

            angle_values[15] = 1 if msg.human[u].body.left_hand_state > 1 else 0
            angle_values[16] = 1 if msg.human[u].body.right_hand_state > 1 else 0
            
            self.motion.setAngles(self.angle_names, angle_values, 0.15)#速度も適応的にかえる

            
def main(args):

    rospy.init_node("pepper_mimic", anonymous=True)

    #topic = rospy.get_param("~topic", "/humans/kinect_v2")
    #rotate = rospy.get_param("~rotate", False)
    
    cp = ConnectPepper()
       
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"
        

if __name__ == "__main__":
    print "start"
    main(sys.argv)
