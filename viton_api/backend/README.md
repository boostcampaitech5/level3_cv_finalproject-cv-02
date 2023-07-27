## 사용한 모델 및 코드 출처
- Preprocess
    - [CarveKit](https://github.com/OPHoperHPO/image-background-remove-tool), [TRACER](https://github.com/Karel911/TRACER) for Masking.
    - [CIHP_PGN](https://github.com/Engineering-Course/CIHP_PGN), [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for Human Parsing.
    - [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for Pose Estimation.
    - [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) for DensePose.

- Virtual Try-On
    - [HR-VITON](https://github.com/sangyun884/HR-VITON)
    - [LaDI-VTON](https://github.com/miccunifi/ladi-vton)


## 디렉토리 구조

```
backend
├── infra : 전처리 / try-on 관련 client 및 모델, inference 코드가 정의되어있는 폴더.
│   ├── preprocess : 전처리 관련 폴더
│   │   ├── densepose/
│   │   ├── human_parser/
│   │   ├── masking/
│   │   └── pose_estimation/
|   |
│   └── viton : try-on 관련 폴더
│       ├── hr_viton/
│       ├── ladi_viton/
│       └── client.py : 모든 viton 모델이 같이 사용하는 client 코드
│
├── routers : 프로그램 별 API가 정의되어있는 폴더입니다.
│   ├── densepose.py : densepose 관련 API를 정의한 코드
│   ├── human_parser.py : SCHP 관련 API를 정의한 코드(CIHP_PGN은 비활성화한 상태)
│   ├── masking.py : masking 관련 API를 정의한 코드(base: tracer_b7)
│   ├── pose_estimation.py : OpenPose 기반 pose estimation 관련 API를 정의한 코드
│   ├── proxy.py : try-on inference 과정에 필요한 모든 API를 정의한 코드
│   └── viton.py : try-on 관련 API를 정의한 코드
│    
├── backend_requirements.txt : pip freeze로 export한 패키지 리스트
├── backend_requirements.yaml : conda env export로 만든 패키지 리스트 (패키지 설치 과정에서 혹시모를 에러를 방지하기 위해 생성)
├── cihp_pgn_requirements.yaml : CIHP_PGN 모델 사용 시 필요한 환경
├── densepose.py : try-on 모델 사용에 필요한 densepose map을 얻기 위해 사용하는 코드
├── human_parser.py : try-on 모델 사용에 필요한 human parse map을 얻기 위해 사용하는 코드
├── masking.py : try-on 모델 사용에 필요한 mask img를 얻기 위해 사용하는 코드
├── pose_estimation.py : try-on 모델 사용에 필요한 pose img, kpts를 얻기 위해 사용하는 코드
├── proxy.py : 모든 전처리 모델을 backend 내에서 선언, 요청하기 위해 사용하는 코드
└── viton.py : try-on 모델 inference를 위해 사용하는 코드
```