#base library
import time
import os
import pandas as pd
from collections import defaultdict

#crawling library
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests

#multi processing
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

#set url
from .utils import setting_url

class BaseCrwaler():
    def __init__(self, config):
        #config 
        self.config = config

        #crawling 정보 저장
        ##linkes의 경우 중복을 막기 위해서 set으로 지정
        self.rank_url = list()
        self.linkes, self.key, self.price, self.info = set(), set(), defaultdict(), []

    def run(self):
        #1. 상품 url crawling
        ## 1.1 크롤링할 상품수 만큼 랭킹 url 리스트로 저장
        self.make_rank_url(self.rank_url, self.config)
 
        ## 1.2 저장된 url 리스트를 활용해서 상품주소 crawling 
        ### 멀티 프로세싱으로 처리
        self.do_thread_crawl(self.scrape_goods_all_url, self.rank_url, self.linkes, self.price, self.key)

        print('-' * 80)            
        print('크롤링 된 상품 수 : ', len(self.linkes))
        print('-' * 80)

        #2. 상품 url에 들어가서 상품 정보 crawling 및 이미지 저장
        
        print('상품 정보를 크롤링 중입니다...')
        ## 멀티 프로세싱으로 처리
        self.do_thread_crawl(self.scrape_main_info, self.linkes, self.info, self.config)

        #3. 다운로드 받은 이미지와 링크 수가 맞는치 check
        print('-' * 80)
        print('누락된 상품 정보를 탐색중입니다...')
        self.check_info(self.linkes, self.key, self.info, self.config)
        print('-' * 80)

        #4. 크롤링한 상품정보 DataFrame으로 저장
        self.make_dataframe( self.linkes, self.key, self.info, self.price, self.config)
        print(f'상품 이미지와 상품정보 csv가 저장되었습니다!! 저장경로 :' + self.config['SAVE_PATH'])

    def scrape_want_page(self, url):
        """_summary_

        Args:
            url (str): url that want to scrapy

        Returns:
            soup (BeautifulSoup) : url page html
        """
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'lxml')
        
        return soup
    
    def make_rank_url(self, rank_url, config):
        """_summary_

        Args:
            rank_url (list): rank url save list
            config (dict): config setting
        """
        start = config['START']
        end = config['END']
        for i in range(start, end+1):
            now_url = setting_url(i, config['CATEGORY'])
            rank_url.append(now_url)
            
    
    def scrape_goods_all_url(self, url, linkes, price, key):
        """_summary_

        Args:
            url (list): scrape rank url
            linkes (list): goods url save list 
            key (list): goods key save list
        """
        soup = self.scrape_want_page(url)

        self.scrape_goods_url(soup, linkes, price, key)
        time.sleep(1)
        
        if len(linkes) % 100 == 0:
            print('현재 크롤링된 linkes 수 : ', len(linkes))


    def scrape_goods_url(self, soup, linkes, price, key):
        """_summary_

        Args:
            soup (html): goods url html
            linkes (list): goods url save list
            key (list): goods key save list
        """
        #a 태그에 img-block class를 불러와서 list로 저장
        
        sorce = soup.find_all('div', attrs={"class":"list-box box"})

        temp_link, temp_price = [], []
        for i in sorce:
            temp_link.append(i.find_all('a', attrs={"class": "img-block"}))
            temp_price.append(i.find_all('p', attrs={"class": "price"}))

        temp_link = temp_link[0]
        temp_price = temp_price[0]

        for idx, element in enumerate(temp_link):
            linkes.add('https:' + element['href'])
            key.add(element['href'].split('/')[-1])

            tp = temp_price[idx].get_text().replace(' ', '').replace('\n', '')
            if len(tp) == 3:
                price[element['href'].split('/')[-1]] = tp.split('원')[1]
            else:
                price[element['href'].split('/')[-1]] = tp.split('원')[0]



    def scrape_main_info(self, url, info, config):
        """_summary_

        Args:
            url (str): good url
            info (list): goods info save list
            config (dict): config setting
        """

        #상품 page에서 주요 정보 crwaling
        good_key = url.split('/')[-1]
        img_path  = os.path.join(config['SAVE_PATH'], 'img', config['CATEGORY'])

        try:
            if not os.path.exists(img_path):
                os.makedirs(img_path)
        except:
            pass

        
        responses = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        time.sleep(1)

        #세부 상품 page로 이동
        soup = BeautifulSoup(responses.text, 'lxml')
        informations = soup.find_all('div', attrs={'class':'product-img'})
        brands = soup.find_all('p', attrs={'class':'product_article_contents'})
        categories = config['CATEGORY']
        subcategories = soup.find_all('p', attrs={'class':'item_categories'})
        meta_data = soup.find_all('a', attrs={'class':'listItem'})

        temp = []
        for infoes in informations:
            for key in infoes:
                try:
                    #img download
                    urlretrieve('https:' + key['src'], os.path.join(img_path, good_key+'.png'))
                    temp.append(good_key)
                    temp.append(key['alt'])
                    temp.append('https:'+ key['src'])
                    temp.append(os.path.join(img_path, good_key+'.png'))
                    
                except:
                    pass

        # 브랜드, 카테고리 저장
        try:
            for f in brands[0]:
                temp.append(f.find('a').string)
        except:
                temp.append(None)

        temp.append(categories)

        # 서브 카테고리 저장
        subcategory = list(f for f in subcategories[0])
        if len(subcategory) == 7:
            temp.append(subcategory[3].string)
        elif len(subcategory) == 9:
            temp.append(subcategory[5].string)
        else:
            temp.append(None)

        #메타 데이터 추출
        if len(meta_data) != 0:
            temp2 = []
            for idx in range(len(meta_data)):
                temp2.append(meta_data[idx].string)
        else:
            temp2 = None

        temp.append(temp2)

        #info list에 수집한 정보 저장
        info.append(temp)


        if len(info) % 50 == 0:
            print('현재 크롤링된 이미지 수 :', len(info))
            
    def check_info(self, linkes, keyes, info, config):
        """_summary_

        Raises:
            Exception: if len(info) > len(linkes) - ERROR
        """
        #linkes의 길이와 info의 길이가 같을때 까지 누락된 정보 탐색 
        if not isinstance(linkes, set):
            raise Exception('linkes가 set이 아닙니다')
        
        if not isinstance(keyes, set):
            raise Exception('key가 set이 아닙니다')
        
        if not isinstance(info, list):
            raise Exception('info가 list가 아닙니다')

        while len(linkes) != len(info):
            if len(linkes) < len(info):
                raise Exception('info의 정보가 linkes 보다 큽니다!!')
            
            leak = [] #누락된 키값 저장
            
            temp_image = [] #이미지 정보 저장
            for values in info:
                temp_image.append(values[1].split('/')[-1])

            for key in keyes:
                if key not in temp_image:
                    leak.append(key)

            if len(leak) == 0:
                print('누락된 정보가 없습니다')
                return 

            print(f'누락된 상품정보 : {leak}')
            print('누락된 상품정보를 다시 크롤링 합니다!!')
            BASE_URL = 'https://www.musinsa.com/app/goods/'

            for key in leak:
                self.scrape_main_info(BASE_URL + key, info, config)

    def make_dataframe(self, linkes, key, info, price, config):
        """_summary_
        save dataframe that made by goods info
        """

        if not os.path.exists(os.path.join(config['SAVE_PATH'], 'csv')):
            os.makedirs(os.path.join(config['SAVE_PATH'], 'csv'))

        df = pd.DataFrame()

        df['key'] = sorted(list(key))
        df['link'] = sorted(list(linkes))

        infoes = sorted(info, key=lambda x:x[0])
        


        #상품이름, 이미지링크, 저장주소, 메타데이터 dataframe에 저장
        temp_name = []
        temp_images = []
        temp_images_path = []
        temp_brand= []
        temp_price = []
        temp_category = []
        temp_subcategory = []
        temp_meta = []

        for idx, info in enumerate(infoes):
            temp_name.append(info[1])
            temp_images.append(info[2])
            temp_images_path.append(info[3])
            temp_brand.append(info[4])
            temp_category.append(info[5])
            temp_subcategory.append(info[6])
            temp_price.append(price[sorted(list(key))[idx]])
            temp_meta.append(info[7])
            
            
        df['name'] = temp_name
        df['image_link'] = temp_images
        df['path'] = temp_images_path
        df['brand'] = temp_brand
        df['price'] = temp_price
        df['category'] = temp_category
        df['subcategory'] = temp_subcategory
        df['meta'] = temp_meta
    
        df.to_csv(os.path.join(config['SAVE_PATH'], 'csv', config['CATEGORY']+str(config['START']) + '_' + str(config['END']) + '.csv'), index=False)

    def do_thread_crawl(self, func, listes, *args):
        """_summary_

        Args:
            func (func): useing function for multiprocessing
            listes (list): func input 
        """
        thread_list = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            for url in listes:
                thread_list.append(executor.submit(func, url, *args))
            
            for execution in concurrent.futures.as_completed(thread_list):
                execution.result()
            
            executor.shutdown()


