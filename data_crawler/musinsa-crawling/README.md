# Musinsa Crawling 

## 01. crawler.py
```
class : BaseCrwaler()
┣ __init__ 
┣ run : crwaling 실행
┣ scrape_want_page : URL의 html scrape
┣ make_rank_url : crwaling 할 상품 수 만큼 rank page url 생성
┣ scrpae_goods_all_url : 생성한 rank page에서 상품 주소 srcape
┣ scrape_goods_url : 개별 상품 페이지 주소 scrape
┣ scrape_main_info : 상품 페이지 주소로 상품정보 scrape
┣ check_info : crwaling 이후 누락된 정보 check
┣ make_dataframe : crwaling한 정보를 dataframe으로 저장
┗ do_thread_crawl : multiprocessing으로 crwaling method 처리
```

## 02. utils.py
- 기본설정 관련 method

```
┣ parse_args : argparser 설정
┣ setting_url : url 설정 
┗ config_setting : config 설정
```

## 03. main.py
`: 크롤링 예시 코드`

```
#custom crawling method
from musinsa import *
from musinsa import BaseCrwaler

#for play time
import time
import datetime

if __name__ == "__main__":
    start = time.time()
    #parser 설정
    args = parse_args()

    #기본 정보 설정
    config = config_setting(args)

    #Crawlwer 설정
    crawler = BaseCrwaler(config)

    print(config)

    #Crawling 
    result = crawler.run()
    end = time.time()

    total_sec = end - start
    result = str(datetime.timedelta(seconds=total_sec)).split('.')
    print(f'crawling에 소요된 시간 : {result[0]}')

```
