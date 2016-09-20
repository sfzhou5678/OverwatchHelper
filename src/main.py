#encoding:utf-8

'''

Created by ZSF on 2016/9/20.

参考资料：
[1] 守望先锋人工智能辅助：硬件驱动和自动稳枪 https://zhuanlan.zhihu.com/p/22404957

[2] Opencv Python版学习笔记（五）光流跟踪 Lucas-Kanade（LK）算法
http://cache.baiducontent.com/c?m=9f65cb4a8c8507ed4fece7631046893b4c4380146d96864968d4e414c42246141f2de1b0203f4344959e2d3956b21f0baca36d2c761e2bb79bcc8240dcafd7756fde28230017913612c418dfdc3726d654954de8df0e96cae74592b9a2d6c82759dd537438cbb6d1075c&p=9865c70385cc43ff57ee947d465ecd&newp=84769a47cd8709ff57ee947d46498f231610db2151d4da106b82c825d7331b001c3bbfb423231107d0c5776c02ad4b5ee9f43274350123a3dda5c91d9fb4c5747993637c&user=baidu&fm=sc&query=%B9%E2%C1%F7%CB%E3%B7%A8+python&qid=c1f1d9c600027f20&p1=2

注：要使用本程序需要安装opencv(略麻烦)

'''

import numpy as np
import cv2

import win32api
import win32con

from PIL import ImageGrab

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self):#构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.frame_idx = 0



    def run(self):#光流运行方法
        # 监视窗口大小
        windowWidth=800
        windowHeight=200

        # 将监视窗口划分为ROW*COL的格子，统计各个格子出现的旅店个数，然后控制鼠标移动
        rowCount=9
        colCount=17

        unitWidth=windowWidth/colCount
        unitHeight=windowHeight/rowCount

        # 准星位置，目前ROW需要手动调整，col为中值
        mousePosRow=6
        mousePosCol=colCount//2+1
        while True:
            ret=True
            frame=ImageGrab.grab(bbox=(100,150,100+windowWidth,150+windowHeight))
            frame = np.array(frame) #this is the array obtained from conversion
            if ret == True:
                dotCounter=np.zeros(colCount*rowCount)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#转化为灰度虚图像
                vis = frame

                if len(self.tracks) > 0:#检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)#前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)#当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)#得到角点回溯与前一帧实际角点的位置变化关系

                    #判断d内的值是否属于[0.001,1]，不数据则被认为是错误的跟踪点
                    d[d<0.001]=10
                    good = d<1

                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):#将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                        targetPos=(y//unitHeight)*colCount+x//unitWidth
                        if(targetPos>=colCount*rowCount or targetPos<0):
                            continue
                        dotCounter[targetPos]+=1

                    # print(dotCounter)
                    # 找到绿点数最多的格子，如果格子不唯一则continue
                    maxPosArray=np.where(dotCounter==dotCounter.max())
                    if(len(maxPosArray[0])==1):
                        if(dotCounter[int(maxPosArray[0])]>=4): # 每个格子中的绿点数要大于某个值，避免枪乱动
                            maxDotPos=int(maxPosArray[0])
                            movementx=(maxDotPos%colCount-mousePosCol)*unitWidth
                            movementy=(maxDotPos//colCount-mousePosRow)*unitHeight

                            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE | win32con.MOUSEEVENTF_MOVE,int(movementx),int(movementy))
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    self.tracks = new_tracks
                    # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))#以上一振角点为初始点，当前帧跟踪到的点为终点划线
                    #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                # if self.frame_idx % self.detect_interval == 0:#每5帧检测一次特征点
                mask = np.zeros_like(frame_gray)#初始化和视频大小相同的图像
                mask[:] = 255#将mask赋值255也就是算全部图像的角点
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:#跟踪的角点画圆
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)#像素级别角点检测
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])#将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

def main():
    App().run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()