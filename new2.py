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

url_facemetric = "http://10.0.7.145:8080"
#api_key = 'd6ad30a3-a69f-464c-a76c-ea0734ebc6b8'
camera_id = 8
create_time = datetime.datetime.now()
update_time = create_time
faces_array_limit = []
limit = 35
faces_dict = {}
time1 = 0

flagg = []

batchi = {
    'camera_id': camera_id,
    'external_id': 0,
    'datetime': create_time,
    'update_time': update_time,
    'batchDuration': 40,
    'faces': [],
    'sap_push': 0
}


def create_batch():
    create_time = datetime.datetime.now()
    update_time = create_time

    batchi['datetime'] = create_time
    batchi['update_time'] = update_time

    return batchi


def send_batch(faces_array, batchi, ids):
    print('\n ========= send batch===========')
    global flagg
    url = f'{url_facemetric}/api/search'

    temp_dict = {'add_after_search': True,
                 'add_limit': 0.0,
                 'similar_images_threshold': 0.75,
                 'min_image_quality': 0.60,
                 'max_yaw': 16,
                 'max_pitch': 16,
                 'min_face_width': 15,
                 'min_face_height': 15}

    # try:
    #print("FACE ARRAY", len(faces_array))
    response = requests.post(url, files=faces_array, params=[
        ('external_id', batchi["external_id"]),
        #('limit', 1000),
        #('api_key', batchi["api_key"]),
        ('add_after_search', temp_dict["add_after_search"]),
        ('min_similarity', temp_dict["similar_images_threshold"]),
        ('min_image_quality', temp_dict["min_image_quality"]),
        ('max_yaw', temp_dict["max_yaw"]),
        ('max_pitch', temp_dict["max_pitch"]),
        ('min_face_width', temp_dict['min_face_width']),
        ('min_face_height', temp_dict['min_face_height']),
        ('camera_id', batchi['camera_id'])
    ])
    #print('*' *20)
    #print(batchi["external_id"])
    # print(response.status_code)
    ttext = json.loads(response.text)
    if response.status_code == 200:
        print('============== отправил c id', ids, datetime.datetime.now(),'===============')
        print(ttext['matches']['items'][:2])
        is_saper = False
        if len(ttext['matches']['items']) > 0: is_saper = 'GUEST' not in ttext['matches']['items'][0]['reference_image']['group']['id']
        if ids in flagg or is_saper:
            print('\n\nновый человек')
            if ids in flagg:
                del flagg[flagg.index(ids)]
            sap.send_push(response.text, faces_array)

            
        else: 
            print('\n\npovtorniy push')
    else:
        print('\nне отправил c id', ids, response.status_code, response.text)
        if ids in flagg: del flagg[flagg.index(ids)]
    print('*'*20)
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
    print(args)
#    cam = cv2.VideoCapture('rtsp://admin:meinsmmeinsm@10.0.7.117:554/stream/profile1')
    #cam = cv2.VideoCapture('rtsp://admin:meinsmmeinsm@10.0.7.117:554/stream/profile1')

    #cam = cv2.VideoCapture('rtsp://admin:meinsmmeinsm@10.0.7.117:554/stream/profile0') 
          # cv2.VideoCapture('http://admin:meinsmmeinsm@10.0.7.117:80/control/faststream.jpg?stream=full&needlength&preview&previewsize=1920x1080&quality=40&fps=12.0&camera=left'))
    #cam = cv2.VideoCapture('output.avi')
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
    print('------SIZE IMAGE', cam.read().shape, '--------')
    c = 0
    ct = CentroidTracker()
    while True:
        flag = 0
        image = cam.read()

        if image is None:
            print('------------BREAK NONE IMG-------------')
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
            print('loop2/checking')
            faces_dict, temp_dict, ct, time1, _ = track2(faces_dict, rects, temp, temp_dict, ct, time1, img)

        for i, (bb, ll) in enumerate(zip(dets, landmarks)):
            print('------- found face ----------')
            e_box = sap.eyes_box(bb, ll)
            temp = sap.read_temp(e_box, therm_img)
            #print('max temperature is', max_eyes_temp)
            
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
            #startX, endX, startY, endY = int(e_box[0][0]), int(e_box[2][0]), int(e_box[2][1]), int(e_box[0][1])

            #face = cv2.resize(face, (224, 224))
            #print('FACE', face.shape, orig_face.shape)
            if face is not None and len(face) > 0:
                print('shapes', orig_face.shape, face.shape)
                #face = cv2.resize(face, (224, 224))
                if orig_face.shape[0] > 90 and orig_face.shape[1] > 90:
                    startX, endX, startY, endY = n_x, x2 + gap_x, n_y, y2 + gap_y
                    rects.append((startX, startY, endX, endY))



        if len(dets) > 0:
            faces_dict, temp_dict, ct, time1, img = track2(faces_dict, rects, temp, temp_dict, ct, time1, img)
        else:
            ct.update(rects)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps * 0.95 + curr_fps * 0.05)
        #if fps > 4.0: print(f'fps ={fps}')
        #print(f'fps ={fps}')

        tic = toc

        # FIXME: удалить после теста
        if len(dets) > 0:
            #print('save')
            #cv2.imwrite('/home/fm/cameraservice-jetson_nano/out_t/images/' + str(c) + '.jpg', img)
            c += 1


def track2(faces_dict, rects, temp, temp_dict, ct, time1, img, limit = 35):
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
        print('FLAGS', flags, 'to push', flagg)        
        for (objectID, face) in faces.items():
            #print('-----track2/face of faces', face) 
            try:
                temp_dict[objectID].append(temp)
                faces_dict[objectID].append(img[face[1]:face[3], face[0]:face[2]])
            except KeyError:
                faces_dict[objectID] = [img[face[1]:face[3], face[0]:face[2]]]
                temp_dict[objectID] = [temp]
                #print('track2/faces_dict', faces_dict)
            #cv2.imwrite('/home/fm/cameraservice-jetson_nano/out_t/images2/' + str(time.time()) + '.jpg', img[face[1]:face[3], face[0]:face[2]])
        #FIXME убрать весь фор после тестов
        print('faces dict', faces_dict.keys(), 'flags', flags.keys())
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
        
        #cv2.imwrite('/home/fm/cameraservice-jetson_nano/out_t/images2/' + str(time.time()) + '.jpg', max_face)
    #print('\n/////////// MAX IMAGE //////////////////////')
    return max_face


def batch(photo, ids):
    #print(f'len(face_list)={len(face_list[0])}')
    #print(f'face list = {face_list}')
    faces_array = []
    #print("\n////// batch ////////\n")
    batchi = create_batch()
    uid = str(uuid.uuid4())
    success, img1 = cv2.imencode(".jpg", photo)
    if img1 is not None:
        faces_array.append(("image", img1))
        batchi['faces'].append(photo)
        send_batch(faces_array, batchi, ids)
    else:
        print('аномалия')
        #print(face_list)


main()
