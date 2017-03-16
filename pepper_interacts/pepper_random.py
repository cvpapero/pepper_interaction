#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2017/1/25
一定の時間でしゃべる


2017/1/19
Randomに話者交代Timingなどをかえる
発話時の動きはAPI
適当に相槌

"""


import argparse
import numpy as np
import json
import os
import copy
import sys
import threading
import time

import rospy
from humans_msgs.msg import Humans
from humans_msgs.msg import AudioData
from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


from naoqi import ALProxy

import rviz_box
import text_example



class PepperInteract():
    def __init__(self):

        rospy.init_node("random_pepper", anonymous=True)
        self.apub = rospy.Publisher('/person_robot/anno', Float64MultiArray, queue_size=10)
        
        robot_ip = "133.19.23.152"
        port = 9559
        self.asp = ALProxy("ALAnimatedSpeech", robot_ip, port)
        #self.config = {"bodyLanguageMode":"contextual"}
        self.config = {"bodyLanguageMode":"random"}
        self.texts = text_example.input_texts()

        self.audiodevice = ALProxy("ALAudioDevice", robot_ip, port)
        self.audiodevice.setOutputVolume(30)#45

        self.stop_pepper()

        self.speak = ALProxy("ALTextToSpeech", "133.19.23.152", 9559)
        self.speak.setLanguage("Japanese")
        self.speaker = False
        
        self.motion = ALProxy("ALMotion",robot_ip, port)
     
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

        self.angle_values=[-0.38950430949660736, 0, 0.45960922469723942, 1.1435471961738974, -1.6492293794923505, -1.5707963267948966, 0, -0.3860913233267792, 1.1308656711455178, 1.6328734238916121, 1.5707963267948966, 0, -0.037573553589435829, -0.065725886252029936, 0, 0, 0]

        self.rmin, self.rmax = 100, 300
        self.max_count = np.random.randint(self.rmin, self.rmax)
        
        threading.Thread(target=self.statePub).start()
        threading.Thread(target=self.timekeeper).start()


    def stop_pepper(self):
        ba = ALProxy("ALBasicAwareness","133.19.23.152",9559)
        ba.stopAwareness()
        am = ALProxy("ALAutonomousMoves","133.19.23.152",9559)
        am.setBackgroundStrategy("none");
        am.setExpressiveListeningEnabled(0);

    #anno Pub
    def statePub(self):

        while not rospy.is_shutdown():

            person_anno = 1 if self.speaker==False else 0
            robot_anno = 1 if self.speaker==True else 0
            
            amsgs = Float64MultiArray()
            amsgs.data.append(person_anno)
            amsgs.data.append(robot_anno)
            self.apub.publish(amsgs)
            time.sleep(0.01)

            
    def timekeeper(self):

        count = 0
        while not rospy.is_shutdown():
            print count,"/",self.max_count,self.speaker  
            if count > self.max_count:
                self.speaker = not self.speaker
                
                if self.speaker == False:
                    #発話を停止
                    self.speak.stopAll()
                    self.motion.stopMove()
                    text = "^start(animations/Stand/Gestures/Yes_2) "
                    self.asp.say(text)
                    print "stop speak"
                   
                self.max_count = np.random.randint(self.rmin, self.rmax)                
                count = 0
                
            count += 1
            
            time.sleep(0.1)
        
            
    def speak_turn(self):
     
        while not rospy.is_shutdown():

            #print "Robot is Speaker:",speaker, count,"/",max_count
            if self.speaker:
                # 話し手なら
                text_idx = np.random.randint(len(self.texts))
                text = self.texts[text_idx]#Randomに選ばれた文章を読上げ
                self.motion.setAngles(self.angle_names, self.angle_values, 0.2)
                self.speak.say(text)
                time.sleep(3)
            else:
                # 聞き手なら
                if np.random.randint(self.max_count/4)==0:
                    text = "^start(animations/Stand/Gestures/Yes_2) はい"
                    self.asp.say(text)
                
                time.sleep(0.1)

 
    

def main():

    pi = PepperInteract()

    #while not rospy.is_shutdown():
    pi.speak_turn()
        
    print "shoutdown"
        
if __name__ == "__main__":
    print "start"
    main()
