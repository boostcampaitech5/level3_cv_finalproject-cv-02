#custom crawling method
from musinsa import *

#for play time
import time
import datetime

if __name__ == "__main__":
    start = time.time()
    config = vton_parse_args()
    Crawling = MusinsaVton(config)
    Crawling.run()

    end = time.time()

    total_sec = end - start
    result = str(datetime.timedelta(seconds=total_sec)).split('.')
    print(f'crawling에 소요된 시간 : {result[0]}')






