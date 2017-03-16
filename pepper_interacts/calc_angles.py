#!/usr/bin/env python
# -*- coding: utf-8 -*-

#https://github.com/malaybaku/KinectForPepper/tree/master/src/KinectForPepper/Models

import numpy as np
import math

class CalcAngles():
    def __init__(self):
        self._x = np.array([1,0,0,0])
        self._y = np.array([0,1,0,0])
        self._z = np.array([0,0,1,0])


    def product_q(self, v, u):
        #http://www.buildinsider.net/small/bookkinectv2/0805
        w = v[3]*u[3] - v[0]*u[0] - v[1]*u[1] - v[2]*u[2]
        x = v[3]*u[0] + v[0]*u[3] + v[1]*u[2] - v[2]*u[1]
        y = v[3]*u[1] - v[0]*u[2] + v[1]*u[3] + v[2]*u[0]
        z = v[3]*u[2] + v[0]*u[1] - v[1]*u[0] + v[2]*u[3]   
        return np.array([x, y, z, w])


    #v: np.array([x,y,z,w])
    def inverse(self, v):
        return np.array([-v[0], -v[1], -v[2], v[3]])

    
    def rotate_by(self, unit, rot):
        c_rot = self.inverse(rot)
        return self.product_q(self.product_q(rot, unit), c_rot)

    
    #shoulder, elbow, wrist: np.array([x,y,z,w])
    def set_arm_angles(self, shoulder, elbow, wrist, arg):

        s_x = self.rotate_by(self._x, shoulder)
        s_y = self.rotate_by(self._y, shoulder)
        s_z = self.rotate_by(self._z, shoulder)
    
        e_z = self.rotate_by(self._z, elbow)
        e_y = self.rotate_by(self._y, elbow)

        e_z_s = np.array([np.dot(e_z.T, s_x), np.dot(e_z.T, s_y), np.dot(e_z.T, s_z), 0])
        e_y_s = np.array([np.dot(e_y.T, s_x), np.dot(e_y.T, s_y), np.dot(e_y.T, s_z), 0])

        w_y = self.rotate_by(self._y, wrist)
    
        s_roll = math.asin(e_y_s[1])
        s_pitch = 0.0
        if s_roll > -1.5 and s_roll < 1.5:
	    if np.fabs(e_y_s[0]) > np.fabs(math.cos(s_roll)):
	        s_pitch = 0.5 * np.pi
	    else:
                if arg == "left":
                    #左手
	            s_pitch = math.asin(-e_y_s[0] / math.cos(s_roll))
                else:
                    s_pitch = math.asin(e_y_s[0] / math.cos(s_roll))
    
	        if e_y_s[2] < 0:
                    s_pitch = np.pi-s_pitch if s_pitch > 0 else -np.pi-s_pitch
                 
        #no_roll_elbow = Quaternion()
        if arg == "left":
            no_roll_e= np.array([-math.cos(s_pitch), 0, -math.sin(s_pitch), 0])
        else:
            no_roll_e= np.array([math.cos(s_pitch), 0, -math.sin(s_pitch), 0])
      
        e_yaw = math.asin(np.dot(self. product_q(no_roll_e, e_z_s), e_y_s))
        e_roll = math.acos(np.dot(w_y, e_y))

        if arg == "left":
            return  -1*s_pitch, s_roll, e_yaw, -1*e_roll

        return -1*s_pitch, -1*s_roll, e_yaw, e_roll




    def set_hip(self, spine):
        s_y = self.rotate_by(self._y, spine)-0.25#若干仰ぎ気味なので
        return math.asin(s_y[2])

    
    def set_head(self, nose, head, spine):
        
        def get_angle(trg_a, trg_b, org):
            #trg=[tx, ty, tz], org=[ox, oy, oz]
            vec_a = trg_a - org
            vec_b = trg_b - org
            cos_t = np.dot(vec_a.T, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
            return np.arccos(cos_t)
        
        # 首の縦回転
        hep = np.pi/2. - get_angle(nose, spine, head)

        # 横回転
        nose_xy = np.array([nose[0], nose[1], 0])
        head_xy = np.array([head[0], head[1], 0])
        nose_xy_org = nose_xy - head_xy
        hey = -np.arctan2(nose_xy_org[1], np.fabs(nose_xy_org[0]))
        
        return hep, hey

    def set_wrist_yaws(self, elbow, wrist, arg):
        
        e_z = self.rotate_by(self._z, elbow)
        w_y = self.rotate_by(self._y, wrist)
        w_x = self.rotate_by(self._x, wrist)
        
        w_x = -1*w_x if arg == "left" else w_x
            
        w_ = np.dot(self.product_q(e_z, w_x), w_y)
        w_yaw = math.asin(w_)

        return w_yaw

    def set_angles_pose(self, rbt):

        def get_angle(trg_a, trg_b, org):
            vec_a = trg_a - org
            vec_b = trg_b - org
            cos_t = np.dot(vec_a.T, vec_b)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_b))
            return np.arccos(cos_t)


        # 首の縦回転     
        nose = np.array([rbt[11*3],rbt[11*3+1],rbt[11*3+2]])
        head = np.array([rbt[3*3],rbt[3*3+1],rbt[3*3+2]])
        spin_shoulder = np.array([rbt[10*3],rbt[10*3+1],rbt[10*3+2]])
        hep = np.pi/2. -  get_angle(nose, spin_shoulder, head)

        
        
        # 首の横回転
        #nose_xy = np.array([rbt[11*3],rbt[11*3+1],0])
        #head_xy = np.array([rbt[3*3],rbt[3*3+1],0])
        #nose_xy_org = nose_xy - head_xy
        #hey = np.arctan2(nose_xy_org[1], np.fabs(nose_xy_org[0]))        
        #self.heys.append(hey)
        #self.heys.pop(0)
        hey = 0 #sum(self.heys)/float(len(self.heys))
        
        
        #Shoulder Pitch
        # left
        shoulder_left_zx = np.array([rbt[4*3],0,rbt[4*3+2]])
        elbow_left_zx = np.array([rbt[5*3],0,rbt[5*3+2]])
        elbow_left_org_zx = elbow_left_zx - shoulder_left_zx
        elo_x_zx = np.fabs(elbow_left_org_zx[0])
        elo_y_zx = elbow_left_org_zx[2]
        lsp = -np.arctan2(elo_y_zx, elo_x_zx)

        
        
        # right
        shoulder_right_zx = np.array([rbt[7*3],0,rbt[7*3+2]])
        elbow_right_zx = np.array([rbt[8*3],0,rbt[8*3+2]])
        elbow_right_org_zx = elbow_right_zx - shoulder_right_zx
        rlo_x_zx = np.fabs(elbow_right_org_zx[0])
        rlo_y_zx = elbow_right_org_zx[2]
        rsp = -np.arctan2(rlo_y_zx, rlo_x_zx)

        

        #脇について(WIP)
        org = np.array([0,0,0])
        shoulder_left = np.array([rbt[4*3],rbt[4*3+1],rbt[4*3+2]])
        elbow_left = np.array([rbt[5*3],rbt[5*3+1],rbt[5*3+2]])       
        lsr = get_angle(elbow_left, org, shoulder_left) - np.pi/2.


        shoulder_right = np.array([rbt[7*3],rbt[7*3+1],rbt[7*3+2]])
        elbow_right = np.array([rbt[8*3],rbt[8*3+1],rbt[8*3+2]])       
        rsr = -(get_angle(elbow_right, org, shoulder_right) - np.pi/2.)

        
     
        shoulder_left = np.array([rbt[4*3],rbt[4*3+1],rbt[4*3+2]])
        elbow_left = np.array([rbt[5*3],rbt[5*3+1],rbt[5*3+2]])
        wrist_left = np.array([rbt[6*3],rbt[6*3+1],rbt[6*3+2]])       
        ler = get_angle(shoulder_left, wrist_left, elbow_left) - np.pi

        
        shoulder_right = np.array([rbt[7*3],rbt[7*3+1],rbt[7*3+2]])
        elbow_right = np.array([rbt[8*3],rbt[8*3+1],rbt[8*3+2]]) 
        wrist_right = np.array([rbt[9*3],rbt[9*3+1],rbt[9*3+2]])  
        rer = np.pi - get_angle(shoulder_right, wrist_right, elbow_right) 

        
        ley = -np.pi/2.
        lwy = 0
  
          
        rey = np.pi/2.
        rwy = 0

        # Hip Roll(wip)
        spin_base_yz = np.array([0,rbt[0*3+1],rbt[0*3+2]])
        spin_mid_yz = np.array([0,rbt[1*3+1],rbt[1*3+2]])
        spin_mid_org_yz = spin_mid_yz - spin_base_yz
        smo_x_yz = np.fabs(spin_mid_org_yz[2])
        smo_y_yz = -spin_mid_org_yz[1]
        hir = np.arctan2(smo_y_yz, smo_x_yz)


        # Hip Pitch
        spin_base_zx = np.array([rbt[0*3], 0 ,rbt[0*3+2]])
        spin_mid_zx = np.array([rbt[1*3], 0 ,rbt[1*3+2]])
        spin_mid_org_zx = spin_mid_zx - spin_base_zx
        smo_x_zx = np.fabs(spin_mid_org_zx[2])
        smo_y_zx = spin_mid_org_zx[0]
        hip = np.arctan2(smo_y_zx, smo_x_zx)-np.pi/24

        
        knp = 0
     
        return hep, hey, lsr, lsp, ler, ley, lwy, rsr, rsp, rer, rey, rwy, hir, hip, knp
        
