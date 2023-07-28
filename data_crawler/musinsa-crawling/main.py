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
    exit()


