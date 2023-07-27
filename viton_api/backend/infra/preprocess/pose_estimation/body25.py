# external-library
import cv2
import numpy as np

# built-in library
import json
import os.path as osp


class OpenPoseBody25():
    def __init__(self):
        super(OpenPoseBody25, self).__init__()
        # BODY_25에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
        self.body_parts = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "MidHip": 8, "RHip": 9,
                        "RKnee": 10, "RAnkle": 11, "LHip": 12, "LKnee": 13, "LAnkle": 14,
                        "REye": 15, "LEye": 16, "REar": 17, "LEar": 18, "LBigToe": 19, 
                        "LSmallToe": 20, "LHeel": 21, "RBigToe": 22, "RSmallToe": 23, "RHeel": 24, "Background": 25}

        self.pose_pairs = [
                        ["Nose", "Neck"], ["Nose", "REye"], ["Nose", "LEye"], ["Neck", "RShoulder"], 
                        ["Neck", "LShoulder"], ["Neck", "MidHip"], ["MidHip", "RHip"], ["MidHip", "LHip"], 
                        ["RHip", "RKnee"], ["LHip", "LKnee"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"], 
                        ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["RKnee", "RAnkle"], ["LKnee", "LAnkle"], 
                        ["REye", "REar"], ["LEye", "LEar"], ["LAnkle", "LHeel"], ["LBigToe", "LHeel"], 
                        ["LSmallToe","LHeel"], ["RAnkle", "RHeel"], ["RBigToe", "RHeel"], ["RSmallToe", "RHeel"]
                    ]
        
        self.color_palette = [
            (51, 0, 153), (102, 0, 153), (153, 0, 102), (0, 51, 153),
            (0, 153, 102), (0, 0, 153), (51, 153, 0), (153, 102, 0),
            (102, 153, 0), (153, 51, 0), (0, 102, 153), (0, 153, 153),
            (0, 153, 102), (0, 153, 0), (153, 153, 0), (204, 0, 0),
            (153, 0, 153), (153, 0, 51), (204, 0, 0), (204, 0, 0),
            (204, 0, 0), (153, 153, 0), (153, 153, 0), (153, 153, 0)
        ]

        # 모델 path (절대 경로여야 에러가 발생하지 않음)
        self.proto_file = "/opt/ml/VIT-ON-Demo/backend/infra/preprocess/pose_estimation/openpose-pytorch/pose_deploy.prototxt"
        self.weights_file = "/opt/ml/VIT-ON-Demo/backend/infra/preprocess/pose_estimation/openpose-pytorch/pose_iter_584000.caffemodel"
 
        # 위의 path에 있는 network 불러오기
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file, self.weights_file)


    def setting_img(self, img_path, mask_path):
        # 저장이름 설정
        save_name = img_path.split('/')[-1].split('.')[0]

        # 이미지 읽어오기
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        # TODO: (256, 192)일 때 estimation이 더 잘 됨 -> kpts 정확도가 vton 생성에 영향이 큰 지 체크해보기
        # 이미지를 고정된 크기로 변환
        img = cv2.resize(src=img, dsize=(384, 512))
        mask = cv2.resize(src=mask, dsize=(384, 512))

        # resize한 이미지의 shape 정보 저장
        img_h, img_w, _ = img.shape
        
        return img, mask, [img_h, img_w, save_name]
    

    def remove_background(self, img_path: str, mask_path: str):
        # 이미지와 마스크 각각의 기본 정보를 불러옴
        img, mask, img_info = self.setting_img(img_path, mask_path)

        # 이미지와 같은 사이즈의 빈이미지 생성
        tmp = np.zeros((img_info[0], img_info[1], 3), np.uint8)

        # 배경제거
        cv2.copyTo(img, mask, tmp)
        img = tmp

        return img, img_info


    def pose_estimation(self, storage: str, img, img_info, model_name: str):
        # network에 넣기위해 전처리
        inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (img.shape[1], img.shape[0]), (0, 0, 0), swapRB=False, crop=False)
        
        # network에 넣어주기
        self.net.setInput(inpBlob)

        # 결과 받아오기
        output = self.net.forward()

        # output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
        H = output.shape[2]
        W = output.shape[3]

        points=[]
        result = []
        for i in range(25):
            probmap = output[0, i, :, :]

            # 최댓값 찾기
            minVal, prob, minLoc, point = cv2.minMaxLoc(probmap)

    
            # 원래 이미지에 맞게 점 위치 변경
            x = (img_info[1] * point[0]) / W
            y = (img_info[0] * point[1]) / H

            # prob가 0.1보다 크면 해당 지점에 포인트가 있다고 가정, 아니면 0으로 처리 
            if prob > 0.1:    
                points.append((int(x), int(y)))
                result.append(round(x, 3))
                result.append(round(y, 3))
                result.append(round(prob, 5))
            else :
                points.append(None)

                for _ in range(3):
                    result.append(0)
        
        # json으로 저장
        json_save_state = self.make_json(storage, result, img_info[2])

        if model_name == 'hr-viton':
            # 검정색 빈 이미지 생성
            empty_img = np.zeros((img_info[0], img_info[1], 3), np.uint8)

            # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
            for pair, color in zip(self.pose_pairs, self.color_palette):
                partA = pair[0]             # Head
                partA = self.body_parts[partA]   # 0
                partB = pair[1]             # Neck
                partB = self.body_parts[partB]   # 1
                
                if points[partA] and points[partB]:
                    cv2.circle(empty_img, (points[partA]), 3, color, thickness=-1, lineType=cv2.FILLED)
                    cv2.circle(empty_img, (points[partB]), 3, color, thickness=-1, lineType=cv2.FILLED)
                    cv2.line(empty_img, points[partA], points[partB], color, 2)


            img_path = osp.join(storage, 'preprocess/pose/img', f'{img_info[2]}.png')
            cv2.imwrite(img_path, empty_img)

            # 유효성 검사
            img_save_state = False
            if osp.exists(img_path) and osp.getsize(img_path):
                img_save_state = True

        if model_name == 'hr-viton':
            return all([json_save_state, img_save_state])       
        else:
            return json_save_state

            
    def make_json(self, storage, pose, save_name):
        ex_json = {
            "version":1.3,
            "people":[
                {
                    "person_id":[-1],
                    "pose_keypoints_2d":[],
                    "face_keypoints_2d":[],
                    "hand_left_keypoints_2d":[],
                    "hand_right_keypoints_2d":[],
                    "pose_keypoints_3d":[],
                    "face_keypoints_3d":[],
                    "hand_left_keypoints_3d":[],
                    "hand_right_keypoints_3d":[]
                }
                    ]
                }
        
        ex_json['people'][0]['pose_keypoints_2d'] = pose

        # 저장
        kpts_path = osp.join(storage, 'preprocess/pose/keypoints', f'{save_name}.json')
        with open(kpts_path, 'w') as f:
            json.dump(ex_json, f, indent=4)

        # 유효성 검사
        save_state = False
        if osp.exists(kpts_path) and osp.getsize(kpts_path):
            save_state = True

        return save_state


    def inference(self, storage_root: str, img_name: str, model_name: str):
        img_path = osp.join(storage_root, 'raw_data/person', img_name)
        mask_path = osp.join(storage_root, 'preprocess/mask/person', img_name)

        # 배경 제거
        img, img_info = self.remove_background(img_path, mask_path)

        # pose 추출
        return self.pose_estimation(storage_root, img, img_info, model_name)
