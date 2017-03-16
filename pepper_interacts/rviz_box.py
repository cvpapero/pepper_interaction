
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import rospy
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

class RvizBox():
    def __init__(self):
        self.llist = [[0, 1, 10, 2, 3, 11], [10, 4, 5, 6], [10, 7, 8, 9]]
        self.carray = self.colors([[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 0, 1], [1, 0.5, 0, 1]])


    def colors(self, clist):
        carray=[]
        for c in clist:
            color = ColorRGBA()
            color.r, color.g, color.b, color.a = c[0], c[1], c[2], c[3]
            carray.append(color)
        return carray

    

    def set_point(self, pos, addx=0, addy=0, addz=0, rotate=False):
        pt = Point()            
        if rotate == True:
            pt.x, pt.y, pt.z = -1*pos[0]+addx, -1*pos[1]+addy, pos[2]+addz
        else:
            pt.x, pt.y, pt.z = pos[0]+addx, pos[1]+addy, pos[2]+addz
        return pt

    
    def rviz_obj(self, obj_id, obj_ns, obj_type, obj_size, obj_color=[0, 0, 0, 0], obj_life=0):
        obj = Marker()
        obj.header.frame_id, obj.header.stamp = "camera_link", rospy.Time.now()
        obj.ns, obj.action, obj.type = str(obj_ns), 0, obj_type
        obj.scale.x, obj.scale.y, obj.scale.z = obj_size[0], obj_size[1], obj_size[2]
        obj.color = obj_color
        obj.lifetime = rospy.Duration.from_sec(obj_life)
        obj.pose.orientation.w = 1.0
        return obj

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
