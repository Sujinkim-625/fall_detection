import os
from pickle import FALSE, TRUE
from textwrap import indent
import cv2
import time
from pandas import array
import torch
import argparse
import numpy as np
import json
from send_alarm import send_alarm_kakao

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

spath='/' #파일 디렉토리 경로
out_path='C:/ret/'

outcount=0 #최종 출력 영상을 위한 변수 

source_list={ #비디오 데이터셋:json파일 딕셔너리 생성 
    'C:\\detection\\detection\\walk_detection\\walk_detection20.mp4':'_test.json'
}

for i in source_list:
    outcount+=1    
    sp=source_list[i].split('/')   
    with open(source_list[i],'r') as j:
        json_data=json.load(j)


def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                    kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


temp = 0.0
cal = 0.0

#알고리즘1에서 사용될 변수. 추후 변경 사항.
t = 0.04 #time for frame : 현재 26초 영상 기준, json 파일 656개 (= 0.39/frame)
v = 0.009 #critical speed : 논문에서 제시된 값.

body8y = 0.0
body11y = 0.0

#알고리즘2에서 사용될 낙상 감지 기준 각도와 변수
degree = 0.78 #45 degree

headx  = 0.0 #0
heady = 0.0
footLx = 0.0 #13
footLy = 0.0
footRx = 0.0 #10
footRy = 0.0

# fall count 용 변수
fall = 0

# 낙상 조건 flag.
# 0 = X, 1 = 조건 1 발동, 2 = 조건 2 발동, 3 = 조건 3 발동.
flag = 0
stable_count = 0

# 웹캠 사용시 알람 발송을 위한 변수
# 1. first_detect : 최초 낙상 감지 Timestamp
first_detect = time.time()

# 2. last_detect : 실시간 낙상 감지 시각 갱신
last_detect = time.time()

# 3. detect_flag : 최초 낙상 감지시 flag up.
# detect flag 가 0일 경우 낙상 감지시 first_detect 갱신 후 flag up.
# detect flag 가 1일 경우 last_detect를 매 프레임 마다 갱신
# last_detect - first_detect = 60 일 경우 알람 발생 1회
# if last_detect - first_detect > 60 and detect_flag is 1
# 후 detect_flag 2로 변환.
# detect_flag 2 일 경우 화면에 "alarm sent" 표시
detect_flag = 0
  

# 조건 1 (condition_one) 에서 쓰일 계산 메소드.
def detect (avg_y, temp):
    global fall
    global t
    #hip의 속도(cal) = (hip 좌표 변화값) / 시간의 변화값
    cal = (abs(avg_y-temp))/t

    #print(avg_y, temp)
    #print("cal(v) : " + str(cal))
    
    #hip의 속도가 critial speed(v) 보다 크다면, 낙상 중(fall)
    if (cal>v):
        #print(cal, v)
        fall += 1
    else:
        fall = 0


# 조건 3 (condition_three) 비율 구하는 함수
def find_ratio(x_list, y_list):
    # 직사각형 = 들어온 좌표들에서(머리 0, 어깨 2 & 5, 발 11 & 14 만 비교)
    # x가 가장 큰 좌표 - 작은 좌표 = 가로 width
    # y가 가장 큰 좌표 - 작은 좌표 = 세로 height
    max_x = max(x_list)
    min_x = min(x_list)
    max_y = max(y_list)
    min_y = min(y_list)

    width = max_x - min_x
    height = max_y - min_y
    ratio = width/height

    return ratio

        
        
        
# 알고리즘 1 - hip speed
def condition_one (joints):
    global fall
    global stable_count
    global flag
    global temp

    if (len(joints['9']) > 0 and len(joints['12']) > 0):

        # 9번(hip right), 12번(hip left)의 y 좌표
        body8y = joints['9'][1]
        body11y = joints['12'][1]

        # (논문 상에서) 시간 t 일 때의 hip의 y좌표는 8번(hip right)과 11번(hip left)의 중간 값.
        avg_y = (body8y + body11y) / 2

        #print("body 8y : " + str(body8y))  # 9번 좌표
        #print("body 11y : " + str(body11y))  # 12번 좌표
        #print("avg : " + str(avg_y))  # hip의 위치

        if (i == 0):
            temp = avg_y
        else:
            detect(avg_y, temp)
            temp = avg_y

        # 낙상 중.
        if (fall > 0 and i != 0):
            print("fall detected (condition 1)")
            fall -= 1
            stable_count = 0
            flag = 1  # flag 1 발동. (다음 condition_two로 현재 프레임 낙상 판별)
        else:
            # print("stable situation")
            stable_count += 1
            flag = 0
            fall = 0

            

        
# 알고리즘 2 - angle between human and ground
def condition_two (joints):
    global fall
    global flag
    global stable_count

    if (len(joints['0']) > 0 and len(joints['11']) > 0
            and len(joints['14']) > 0):
    
        #머리 좌표(head x, head y) + 양 발의 좌표 (foot L xy , foot R xy)
        headx = joints['0'][0]
        heady = joints['0'][1]

        footRx = joints['11'][0]
        footRy = joints['11'][1]
        footLx = joints['14'][0]
        footLy = joints['14'][1]

        #양 발 중간 좌표
        footx = abs(footRx + footLx) /2
        footy = abs(footRy + footLy) /2

        sum = abs(heady - footy) / abs(headx - footx)
        
        
        #논문에서 angle 구하는 값. 
        arctan = np.arctan(sum)

        """
        print("headx : " + str(headx))
        print("heady : " + str(heady))
        print("footx : " + str(footx))
        print("footy : " + str(footy))

        print("sum : " + str(sum))
        print("angle : " + str(arctan))
        """
    print(arctan)
    #일정 각도보다 기울어진 경우, 낙상 판단.
    if (arctan < degree or arctan > 1.2):
        print("fall detected (condition 2)")
        stable_count = 0
        flag = 2 
    else:  # fall 조건 불만족 카운팅
        stable_count += 1
        flag = 0
        fall = 0

    
   

# 알고리즘 3 - body ratio check
def condition_three(joints):
    global flag
    global stable_count
    global fall

    is_fall = False

    # 사각형을 판정하는데 사용되는 관절들. 머리0 , 어깨2 5, 발10 13
    x = [joints['0'][0], joints['2'][0], joints['5'][0], joints['11'][0], joints['14'][0]]
    y = [joints['0'][1], joints['2'][1], joints['5'][1], joints['11'][1], joints['14'][1]]


    # 비율 구하기
    ratio = find_ratio(x, y)

    # flag = 0 #마지막 알고리즘 확인 후, 다음 프레임은 다시 1번 condition 부터 체크
    
    # 가로세로 비율이 threshold(1) 보다 클 시 fall 카운트 + 1
    if ratio > 1:
        fall += 1
        stable_count = 0
        is_fall = True
        print("fall detected (condition 3) during " + str(fall)+ " frames")
      

    # stable count = fall이 detect되었어도 그 뒤로 5프레임 연속으로 fall detect 안될시 안전한 것으로 판단.
    else:
        stable_count += 1
        fall = 0

    # 연속으로 fall detect 안될 경우 fall = 0 으로 복귀
    if stable_count == 5:
        print("return to stablized condition\n")
        fall = 0
        flag = 0
        stable_count = 0

    # 연속 5 프레임동안 fall detect 시 
    elif fall == 5:
        print("##### FALL DETECTED in 5 frames #####\n")
        fall = 0
        stable_count = 0

    return is_fall

# 알고리즘 4 - stand up after fall , 낙상 이후 일어서는지 판단.
def condition_stand_up(joints):
    global flag
    global stable_count
    global fall

    x = [joints['0'][0], joints['2'][0], joints['5'][0], joints['11'][0], joints['14'][0]]
    y = [joints['0'][1], joints['2'][1], joints['5'][1], joints['11'][1], joints['14'][1]]

    # 조건 2 체크
    if(len(joints['0']) >0 and len(joints['11']) >0
           and len(joints['14']) > 0):
        headx = x[0]    # joints['0'][0]
        heady = y[0]    # joints['0'][1]

        footRx = x[3]   # joints['11'][0]
        footRy = y[3]   # joints['11'][1]
        footLx = x[4]   # joints['14'][0]
        footLy = y[4]   # joints['14'][1]

        footx = abs(footRx + footLx)/2
        footy = abs(footRy + footLy)/2

        sum = abs(heady - footy) / (abs(headx - footx))
        arctan = np.arctan(sum)

    # 조건 3 체크
    ratio = find_ratio(x, y)

    if arctan < degree and ratio < 1 :  # 조건2 및 조건3 해소시 stable_count + 1
        stable_count += 1
    else :                              # 조건 해소 실패시 fall +, stable_count는 다시 0으로
        stable_count = 0

    if stable_count == 5: # 연속 5프레임동안 일어설 경우 일어섬 판정
        print("Return to stable situation")
        stable_count = 0
        fall = 0
        flag = 0


# 15번 부터 고려안함
def replace_coco(coco):
    coco = np.concatenate((coco, np.expand_dims((coco[1, :] + coco[2, :]) / 2, 0)), axis=0)

    new_coco = []
    for i in range(14):
        new_coco.append([0.0, 0.0, 0.0])

    new_coco[0] = coco[0]
    new_coco[1] = coco[13]
    new_coco[2] = coco[2]
    new_coco[3] = coco[4]
    new_coco[4] = coco[6]
    new_coco[5] = coco[1]
    new_coco[6] = coco[3]
    new_coco[7] = coco[5]
    new_coco[8] = coco[8]
    new_coco[9] = coco[10]
    new_coco[10] = coco[12]
    new_coco[11] = coco[7]
    new_coco[12] = coco[9]
    new_coco[13] = coco[11]

    return new_coco


#15번 부터 고려안함
def coco_to_body25(coco):

    body25 = dict()
    body25['part_candidates'] = []

    coco = replace_coco(coco)


    #0~7 까지 같음
    for i in range(8):
        body25[str(i)] = coco[i]

    for i in range(8, 25):
        body25[str(i)] = [0.0,0.0,0.0]

    body25['9'] = coco[8]
    body25['10'] = coco[9]
    body25['11'] = coco[10]
    body25['12'] = coco[11]
    body25['8'] = [(body25['9'][0]+body25['12'][0])/2.0 , (body25['9'][1]+body25['12'][1])/2.0, body25['9'][2]]
    body25['13'] = coco[12]
    body25['14'] = coco[13]
    body25['19'] = coco[13]
    body25['20'] = coco[13]
    body25['21'] = coco[13]
    body25['22'] = coco[10]
    body25['23'] = coco[10]
    body25['24'] = coco[10]
    return body25



def fall_down_detection(frame_num, coco):   
    is_fall = False

    print(str(frame_num+1) + "'s frame")

    joints = coco_to_body25(coco)
    
    #5 프레임동안 3가지 조건을 모두 만족시켜야 낙상 감지 (fall +1) , fall=5 : 낙상 판단.
    #1번 조건 (condition_one) 을 만족한다면 flag 값을 이용해, 다음 조건을 판별하도록 진행.
    #마지막 조건 (condition_stand_up)은 매 프레임 확인하여, 5프레임 연속 안정된 상태 (stable)라면 일어섬 판정.
        
    ########## 1 - hip speed
    if flag == 0:
        condition_one(joints)

    ########## 2 - angle between human and ground
    if flag == 1: #condition_one에서 fall detected 된 경우
        condition_two(joints)

    ########## 3 - external rectangle ratio
    if flag ==2 : #condition_two에서 fall detected 된 경우
        is_fall = condition_three(joints)

    
    ########## 4 - judging stand up or not  #if flag == 3: 
    condition_stand_up(joints)

    return is_fall

if __name__ == '__main__':
    
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=i,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=768,   # 영상 크기
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',  # 탐지 박스
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default=out_path+'ret.avi', #적용된 영상 결과 만들기 
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=100000, preprocess=preproc).start() #큐사이즈 5만 -> 10만
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]


    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'DIVX')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0

    ## Temp!
    time.sleep(5)

    is_fall = False
    # Predict Actions of each track.
    while cam.grabbed():
        f=f+1
        frame = cam.getitem()
        # image = frame.copy()

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]
            
            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)


        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)
        # print(f, detections[0].keypoints)
        if detected is not None and len(detections) is not 0:
            #! 0번 개체만 고려, TEMP
            is_fall = fall_down_detection(f, detections[0].keypoints)
            area_max = 0
            if is_fall is True:
                if detect_flag is 0:
                    first_detect = time.time()
                    last_detect = time.time()
                    detect_flag = 1
                elif detect_flag is 1:
                    last_detect = time.time()
                    if last_detect - first_detect > 10:
                        send_alarm_kakao()
                        detect_flag = 2
                elif detect_flag is 2:
                    print("alarm has sent")
            else:
                if detect_flag is 1 or detect_flag is 2:
                    detect_flag = 0


            for i, track in enumerate(tracker.tracks):
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                bbox = track.to_tlbr().astype(int)
                center = track.get_center().astype(int)

                action = 'pending..'
                clr = (0, 255, 0)
                action_name = 'none'


                if is_fall:
                    action_name = "Fall Down"
                    action = '{}'.format(action_name)
                    clr = (255, 0, 0)
                else:
                    action_name = "None"
                    action = '{}'.format(action_name)
                    clr = (255, 200, 0)

                # VISUALIZE.
                if track.time_since_update == 0:
                    if args.show_skeleton:
                        frame = draw_single(frame, track.keypoints_list[-1])

                    # 사람 바운드 박스
                    v1_x = bbox[0]
                    v1_y = bbox[1]

                    v2_x = bbox[2]
                    v2_y = bbox[3]

                    xl = v1_x-v2_x
                    yl = v1_y-v2_y

                    area = abs(xl*yl)


                    if(area_max < area):
                        area_max = area

                # VISUALIZE##################
                for i, track in enumerate(tracker.tracks):
                    if not track.is_confirmed():
                        continue
                    v1_x = bbox[0]
                    v1_y = bbox[1]

                    v2_x = bbox[2]
                    v2_y = bbox[3]

                    xl = v1_x - v2_x
                    yl = v1_y - v2_y

                    area = abs(xl * yl)

                    if (area_max != area):
                        continue

                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)

                    ## coco 스켈레톤 번호
                    coco = detections[0].keypoints
                    coco = replace_coco(coco)

                    # 시각화 하기 !숙제
                    body25 = coco_to_body25(detections[0].keypoints)

                    for j in range(0, 25):
                        # pos = body25[str(j)][0].astype(int)
                        # pos = coco[j].astype(int)
                        org = (int(body25[str(j)][0]), int(body25[str(j)][1]))

                        frame = cv2.putText(frame, str(j), org, cv2.FONT_HERSHEY_COMPLEX,
                                                0.4, (255, 0, 0), 2)
                    if detect_flag is 2:
                        frame = cv2.putText(frame, "alarm sent", (bbox[0] + 50, bbox[1] - 50), cv2.FONT_HERSHEY_COMPLEX,
                                            0.4, clr, 1)
                    ## "Fall Down" or "None"
                    frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                            0.4, clr, 1)

        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Clear resource.
    print("cam.grabbed()", cam.grabbed())

    cam.stop()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
