# 1. 사용방법

- 크롤링한 학습 데이터는 ./data/에 위치

- 그리고 크롤링한 csv의 이름을 `total.csv`로 수정하거나 ./dataset.py의 InferenceDataset에 `self.csv_path`에서 이름을 수정


- 만약 크롤링한 csv가 여러개 라면 하나의 csv로 합쳐주어야 한다

- 그리고 Pretrained 된 모델을 ./save/pretrained 폴더에 위치시키고
- `--pretrained_model_name`과 `--model_name`을 해당 모델에 맞게 변경시켜주면된다

- Run inference.py -> `result.csv` : save good or not -> ['check'] col

## 2. Pretrained model link
[☑️ Pretrained Model Down link](https://bottlenose-oak-2e3.notion.site/Good-or-not-e493966734b9482fbbd92a2073398309?pvs=4)
