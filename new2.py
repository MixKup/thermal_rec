import os 
import time 
import argparse 
import datetime 
import cv2 
import requests 
import numpy as np
# import pycuda.autoinit
import json
import sys
import logging
import uuid
import sap
from scipy.spatial import distance as dist
from collections import OrderedDict
from centroidtracker2 import CentroidTracker
from utils.camera import add_camera_args, Camera
from utils.mtcnn import TrtMtcnn

url_facemetric = ""

camera_id = 8
create_time = datetime.datetime.now()
update_time = create_time
faces_array_limit = []
limit = 35
faces_dict = {}
time1 = 0

flagg = []

batchi = {

}


def create_batch():
    """
    creating batch for send 

    # output:
        batchi - json w/ parameters
    """
    create_time = datetime.datetime.now()
    update_time = create_time

    batchi['datetime'] = create_time
    batchi['update_time'] = update_time

    return batchi


def send_batch(faces_array, batchi, ids):
    global flagg
    url = f'{url_facemetric}'

    temp_dict = {}

    response = requests.post(url, files=faces_array, params=[])
    ttext = json.loads(response.text)
    if response.status_code == 200:        is_saper = False
        if len(ttext['matches']['items']) > 0: is_saper = 'GUEST' not in ttext['matches']['items'][0]['reference_image']['group']['id']
        if ids in flagg or is_saper:
            if ids in flagg:
                del flagg[flagg.index(ids)]
            sap.send_push(response.text, faces_array)

            
        else: 
            print('\n\npovtorniy push')
    else:
        if ids in flagg: del flagg[flagg.index(ids)]

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized ')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')
        sys.exit(-1)

    mtcnn = TrtMtcnn()
    sap.create_logger('l', 'logs', 'log')
    faces_array_limit = []
    loop_and_detect2(cam, mtcnn, minsize=60)



def loop_and_detect2(cam, mtcnn, minsize):
    full_scrn = False
    fps = 0.0
    tic = time.time()
    time1 = time.time()
    faces_dict = {}
    temp_dict = {}
    prev_centers = 0
    temp = 36.6
    frame_count = 0
    c = 0
    ct = CentroidTracker()
    while True:
        flag = 0
        image = cam.read()

        if image is None:
            break
        (h, w) = image.shape[:2]
        img = image[:, :w//2]
        therm_img = image[:, w//2:]
        img = cv2.resize(img, (1920, 1080))
        therm_img = cv2.resize(therm_img,(1920, 1080))
        #print(img.shape)
        #img = img[200:-200, :]
        #therm_img = therm_img[200:-200, :]
        list_of_centers = []
        list_of_faces = []
        dets, landmarks = mtcnn.detect(img, minsize)
        rects = []
        max_eyes_temp = None
        if time.time() - time1 > 15:
            faces_dict, temp_dict, ct, time1, _ = track2(faces_dict, rects, temp, temp_dict, ct, time1, img)

        for i, (bb, ll) in enumerate(zip(dets, landmarks)):
            e_box = _.eyes_box(bb, ll)
            temp = _.read_temp(e_box, therm_img)
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            center_x = (x2 + x1) / 2
            center_y = (y2 + y1) / 2
            centers = [center_x, center_y]
            w = int(x2 - x1)
            h = int(y2 - y1)
            gap_y = int((y2 - y1) / 2)
            gap_x = int((x2 - x1) / 2)
            orig_face = img[y1:y2, x1:x2]
            n_y = y1 - gap_y if y1 - gap_y >= 0 else 0
            n_x = x1 - gap_x if x1 - gap_x >=0 else 0
            face = img[n_y:y2 + gap_y, n_x:x2 + gap_x]

            if face is not None and len(face) > 0:
                print('shapes', orig_face.shape, face.shape)
                if orig_face.shape[0] > 90 and orig_face.shape[1] > 90:
                    startX, endX, startY, endY = n_x, x2 + gap_x, n_y, y2 + gap_y
                    rects.append((startX, startY, endX, endY))



        if len(dets) > 0:
            faces_dict, temp_dict, ct, time1, img = track2(faces_dict, rects, temp, temp_dict, ct, time1, img)
        else:
            ct.update(rects)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        tic = toc



def track2(faces_dict, rects, temp, temp_dict, ct, time1, img, limit = 35):
    """ test version of tracking

    """
    global flagg

    
    if len(faces_dict) > limit or time.time() - time1 > 13:
        if len(faces_dict) > 0:
            print('limit time')
            keys = faces_dict.keys()
            for ids in keys:
                print('попытка отправить id', ids)
                max_img = max_image(faces_dict[ids])
                if max_img is not None:
                    batchi['external_id'] = np.mean(temp_dict[ids])
                    print('=== MEAN TEMP', batchi['external_id'], '=====')
                    batch(max_img, ids)
                else: print('NAAAAAN')
                    
            faces_dict = {}
        time1 = time.time()
    else:
        objects, faces, flags = ct.update(rects) # faces = {id: box, ...};   flags = {id: 0, id: 1, ...}
        for ids in flags.keys():
            if flags[ids] == 1 and ids not in flagg: flagg.append(ids)      
        for (objectID, face) in faces.items():
            try:
                temp_dict[objectID].append(temp)
                faces_dict[objectID].append(img[face[1]:face[3], face[0]:face[2]])
            except KeyError:
                faces_dict[objectID] = [img[face[1]:face[3], face[0]:face[2]]]
                temp_dict[objectID] = [temp]
        for (objectID, centroid) in objects.items():
            if not ct.disappeared[objectID]:
                text = "ID {}".format(objectID)
                #cv2.putText(img, text, (centroid[0] - 10, centroid[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return faces_dict, temp_dict, ct, time1, img

def max_image(faces_list):
    max_face, max_square = None, 0
    for face in faces_list:
        #print('max_image/face', face)
        square = face.shape[0]*face.shape[1]
        if square > max_square and square > 6000:
            max_face, max_square = face, square
    return max_face


def batch(photo, ids):
    faces_array = []
    batchi = create_batch()
    uid = str(uuid.uuid4())
    success, img1 = cv2.imencode(".jpg", photo)
    if img1 is not None:
        faces_array.append(("image", img1))
        batchi['faces'].append(photo)
        send_batch(faces_array, batchi, ids)

main()
