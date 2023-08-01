#base library
import json
from collections import defaultdict
import os
import pandas as pd
import re
from pprint import pprint

#crawling library
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
import requests

#multi processing
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .utils import vton_setting_url, vton_parse_args


class MusinsaVton():
    def __init__(self, config):
        self.category = {
            'Top' : '상의',
            'Pants' : '바지',
            'Skirt' : '스커트',
            '상의': 'Top',
            '바지': 'Pants',
            '스커트': 'Skirt'
        }

        # config
        self.config = config

        #수집할 카테고리
        self.ok_category = list(map(lambda x: self.category[x], config.category))

        #crawling 정보 저장
        self.page_url = list() # Crawling [CodyShop or BrandShop] page url list

        #model page URL
        self.model_url = list()

        #저장할 Goods URL
        self.good_url = list() #ex [[Model 사진 번호, 상품번호, 상품카테고리, 상품이미지 url]] 

        #최종 info
        self.final_url = list()

        self.c = 0
        self.down_load_num = 0

            
    def run(self):
        # 1. page url 만들기
        self.make_page_url(self.page_url, self.config)
        print('-'*80)
        # 2. model url 만들기
        self.do_thread_crawl(self.make_model_url, self.page_url, self.model_url, self.config.type)
        print('-'*80)
        print('크롤링 된 Models page : ', len(self.model_url))
        print('-'*80)

        # 3. good url 만들기
        self.do_thread_crawl(self.make_good_url, self.model_url, self.good_url, self.config.type)
        print('-'*80)
        print('크롤링 된 Goods page : ', len(self.good_url))
        print('-'*80)


        # 4. 상품 정보 및 이미지 주소 저장
        self.do_thread_crawl(self.make_img_url, self.good_url, self.final_url)
        print('-'*80)
        print('이미지 링크 수집 완료 : ', len(self.final_url))
        print('-'*80)


        # 5. 이미지 다운 및 정보 저장
        self.do_thread_crawl(self.down_load_img, self.final_url)
        print('-'*80)
        print('이미지 다운로드 완료')
        print('-'*80)

    def scrape_want_page(self, url):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'lxml')

        return soup
    
    def make_page_url(self, page_url, config):

        start = config.start
        end = config.end

        for idx in range(start, end):
            url = vton_setting_url(idx, config)
            page_url.append(url)

    def make_model_url(self, url, model_url, types):
        soup = self.scrape_want_page(url)
    
        
        if types == 'BrandSnap':
            sorce = soup.find_all('div', attrs={'class': 'articleImg'})
            for i in sorce:
                elms = i.select('a')
                for j in elms:
                    #model 사진 번호 
                    model_num = j['href'].split('/')[-1].split('?p=')[0]

                    #model 번호를 기준으로 model 페이지 url 생성
                    temp_url = f'https://www.musinsa.com/mz/brandsnap/view/{model_num}'

                    #self.model_url에 append
                    model_url.append(temp_url)

                    if len(model_url) % 500 == 0:
                        print('크롤링 되고있는 Model page : ', len(model_url))
                        
          
        elif types == 'Codishop':
            sorce = soup.find_all('a', attrs={'class': 'style-list-item__link'})
            for i in sorce:
                #model 사진 번호 
                model_num = re.sub(r'[^0-9]', '', i['onclick'])

                #model 번호를 기준으로 model 페이지 url 생성
                temp_url = f'https://www.musinsa.com/app/styles/views/{model_num}'

                #self.model_url에 append
                model_url.append(temp_url)

                if len(model_url) % 500 == 0:
                    print('크롤링 되고있는 Model page : ', len(model_url))
                    


    
    def make_good_url(self, url, good_url, types):
        model_num = url.split('/')[-1]
        
        soup = self.scrape_want_page(url)

        """good_url list in
        
            {model_num, model_img_url, good_num, img_url_list}
        """

        if types == 'BrandSnap':
            model_sorce = soup.find_all('img', attrs={'class': 'view-photo'})
            
            model_img_url = model_sorce[0]['src']
            
            good_sorce = soup.find_all('div', attrs={'class': 'related-box box bt'})

            for i in good_sorce:
                scripts = i.find_all('script')
                for scrip in scripts:
                    temp = list(f.replace('\n', '').replace('\t', '') for f in scrip.text.split('{'))
                    
                    for info in temp:
                        if 'goods_no' in info:
                            try:
                                goods =list(int(f.replace("'", '')) for f in info[23:].split(',goods_options')[0].split(': ')[1].split(','))
                                
                                for good in goods:
                                    url = f'https://www.musinsa.com/app/goods/{good}'

                                    temp_info = {
                                        'model_id' : model_num, 
                                        'model_img_url' : model_img_url,
                                        'good_id' : good,
                                        'good_url' : url,
                                    }

                                    good_url.append(temp_info)
                                                        
                                    if len(good_url) % 100 == 0:
                                        print('크롤링 되고있는 Goods page : ', len(good_url))       
                                               
                            except:
                                pass
                                    
        elif types == 'Codishop':
            model_sorce = soup.find_all('div', attrs={'class' :'detail_slider_wrap'})
            for tag in model_sorce:
                model_img_url = 'https:' + str(tag.find_all('img', attrs={'class' : 'detail_img'})[0]['src'])
                
            good_sorce = soup.find_all('a', attrs={'class' : 'styling_img'})
            for tag in good_sorce:
                try:
                    url = 'https://www.musinsa.com/' + tag['href'][:-1]
                    
                    temp_info = {
                            'model_id' : model_num, 
                            'model_img_url' : model_img_url,
                            'good_id' : tag['href'][:-1].split('/')[-2],
                            'good_url' : url,
                            }

                    good_url.append(temp_info)

                    if len(good_url) % 100 == 0:
                        print('크롤링 되고있는 Goods page : ', len(good_url))
                except:
                    pass

    def make_img_url(self, good_url_element, final_url):
            soup2 = self.scrape_want_page(good_url_element['good_url'])
            category_tag = soup2.find_all('p', attrs={'class':'item_categories'})

            try: 
                good_category = list(f for f in category_tag[0])[1].string

                if good_category in self.ok_category:
                    goods_list = soup2.find_all('ul', attrs={'class':'product_thumb'})[0].find_all('img')

                    temp_img_url_list = []
                    for tag in goods_list:
                        img_link = 'https:' + tag['src'].replace('60.jpg', '500.jpg')
                        temp_img_url_list.append(img_link)

                    temp_info = {
                        'model_id' : good_url_element['model_id'], 
                        'model_img_url' : good_url_element['model_img_url'],
                        'good_id' : good_url_element['good_id'],
                        'good_url' : good_url_element['good_url'],
                        'good_img_url' : temp_img_url_list,
                        'category' : good_category
                    }

                    final_url.append({
                        'key': good_url_element['model_id'],
                        'info': temp_info
                    })

                    if len(final_url) % 500 == 0:
                        print('크롤링 되고있는 image_info : ', len(final_url))
            except:
                pass

            self.c += 1

            if self.c % 500 == 0:
                print('진행률 :', self.c, '/', len(self.good_url))

    def down_load_img(self, info):
        info = info['info']
        model_num = info['model_id']
        img_list = info['good_img_url'] 

        save_path = os.path.join(self.config.save_path, 'vton', self.config.type, self.category[info['category']], info['model_id'])
        model_save = os.path.join(self.config.save_path, 'vton', self.config.type, 'model')

        # 저장경로 없으면 생성
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(model_save, exist_ok=True)

        # 모델 이미지 다운
        try:
            urlretrieve(info['model_img_url'], os.path.join(model_save, f'{model_num}.png'))
            for idx, img in enumerate(img_list):

                # 상품 이미지 다운
                urlretrieve(img, os.path.join(save_path, f'{model_num}_{idx}.png'))
                self.down_load_num += 1
        
        except:
            pass
        

        if self.down_load_num % 10 ==0:
            print(f'현재 다운 받은 상품 수 : ', self.down_load_num)

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




