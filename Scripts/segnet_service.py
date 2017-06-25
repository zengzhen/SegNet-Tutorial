#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os.path
import scipy
import argparse
import math
import cv2
import sys
import time

from segnet_demo.srv import *
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

sys.path.append('/usr/local/lib/python2.7/site-packages')
# Make sure that caffe is on the python path:
caffe_root = '/home/logan/workspace/caffe-segnet-cudnn5/'
sys.path.insert(0, caffe_root + 'python')
import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--colours', type=str, required=True)

class image_segmenter:
    
    def __init__(self, args):
        self.bridge = CvBridge()
        
        caffe.set_mode_gpu()
        self.net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

        self.input_shape = self.net.blobs['data'].data.shape
        self.output_shape = self.net.blobs['argmax'].data.shape
        
        self.label_colours = cv2.imread(args.colours).astype(np.uint8)

    def handle_image_segmentation(self, req):
        start = time.time()
        caffe.set_mode_gpu()
        
        try:
            frame = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
        except CvBridgeError as e:
            print(e)
        end = time.time()
        print '%30s' % 'parse request image in ', str((end - start)*1000), 'ms'

        start = time.time()
        frame = cv2.resize(frame, (self.input_shape[3],self.input_shape[2]))
        input_image = frame.transpose((2,0,1))
        #input_image = input_image[(2,1,0),:,:] # May be required, if you do not open your data with opencv
        input_image = np.asarray([input_image])
        end = time.time()
        print '%30s' % 'Resized image in ', str((end - start)*1000), 'ms'

        start = time.time()
        out = self.net.forward_all(data=input_image)
        end = time.time()
        print '%30s' % 'Executed SegNet in ', str((end - start)*1000), 'ms'

        start = time.time()
        segmentation_ind = np.squeeze(self.net.blobs['argmax'].data)
        segmentation_ind = cv2.resize(segmentation_ind, (640, 480), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        segmentation_ind_3ch = np.resize(segmentation_ind,(3, 480, 640))
        #segmentation_ind_3ch = np.resize(segmentation_ind,(3,self.input_shape[2],self.input_shape[3]))
        segmentation_ind_3ch = segmentation_ind_3ch.transpose(1,2,0).astype(np.uint8)
        segmentation_rgb = np.zeros(segmentation_ind_3ch.shape, dtype=np.uint8)

        print(self.label_colours)
        cv2.LUT(segmentation_ind_3ch, self.label_colours,segmentation_rgb)
        segmentation_rgb = segmentation_rgb.astype(float)/255

        end = time.time()
        print '%30s' % 'Processed results in ', str((end - start)*1000), 'ms\n'

        #cv2.imshow("Input", frame)
        cv2.imshow("SegNet", segmentation_rgb)
        
        #segmentation_rgb = (segmentation_rgb*255).astype(np.uint8)
        #cv2.imwrite("example_result.png", segmentation_rgb)
        
        key = cv2.waitKey(100)
        
        resp = ImageSegmentationResponse()
        resp.segment = self.bridge.cv2_to_imgmsg(segmentation_ind, "mono8")
        return resp

    def start_server(self):
        rospy.init_node('segnet_demo')
        s = rospy.Service('image_segmentation', ImageSegmentation, self.handle_image_segmentation)
        print "Ready to segment image"
        rospy.spin()


if __name__ == "__main__":
    #--model Example_Models/segnet_sun.prototxt --weights Example_Models/segnet_sun.caffemodel --colours Scripts/sun.png --dataset lobby_test1
    args = parser.parse_args()

    #cv2.namedWindow("Input")
    cv2.namedWindow("SegNet")
    
    seg = image_segmenter(args)
    seg.start_server()

    cv2.destroyAllWindows()

