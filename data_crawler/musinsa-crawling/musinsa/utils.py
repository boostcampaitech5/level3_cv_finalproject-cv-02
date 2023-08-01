#base library
from collections import defaultdict
from argparse import ArgumentParser

def parse_args():
    """_summary_

    Returns:
        args(parser) : base parser setting 
    """
    parser = ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./save/', help='Crawling data save path')
    parser.add_argument('--category', type=str, default='Pants', help='Goods category')
    parser.add_argument('--start', type=int, default=91, help='start page')
    parser.add_argument('--end', type=int, default=100, help='end page')

    args = parser.parse_args()
    return args

def setting_url(idx, category):
    """_summary_

    Args:
        idx (int): setting page index
        category (str): goods category

    Returns:
        BASE_URL (str): changed page url
    """

    url_dict = {
        'Top': '001',
        'Outer': '002',
        'Pants': '003',
        'Onepiece': '020',
        'Skirt': '022'
    }

    BASE_URL = f'https://www.musinsa.com/categories/item/{url_dict[category]}?d_cat_cd=001&brand=&list_kind=small&sort=pop_category&sub_sort=&page={str(idx)}&display_cnt=90&group_sale=&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&plusDeliveryYn=&kids=&color=&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure=/'

    return BASE_URL

def config_setting(args):
    """_summary_

    Args:
        args (parser): base parser setting

    Returns:
        config (defaultdict): config info
    """
    config = defaultdict()

    config['URL'] = setting_url(1, args.category)
    config['START'] = args.start
    config['END'] = args.end
    config['CATEGORY'] = args.category
    config['SAVE_PATH'] = args.save_path

    return config


def vton_parse_args():
    
    parser = ArgumentParser()

    parser.add_argument('--save_path', type=str, default='./save/', help='Save data path')
    parser.add_argument('--type', type=str, default='BrandSnap', help='What type Dataset Crawling ? [Codishop, BrandSnap]')
    parser.add_argument('--category', type=list, default=['Top', 'Pants'], help='What category do you Crawling?')
    parser.add_argument('--start', type=int, default=50, help='start page')
    parser.add_argument('--end', type=int, default=60, help='end page')

    args = parser.parse_args()
    return args

def vton_setting_url(idx, args):

    if args.type == 'Codishop':
        URL = f'https://www.musinsa.com/app/styles/lists?use_yn_360=&style_type=&brand=&model=&tag_no=&max_rt=&min_rt=&display_cnt=60&list_kind=big&sort=date&page={idx}'
    
    elif args.type == 'BrandSnap':
        URL = f'https://www.musinsa.com/mz/brandsnap?_m=&gender=&mod=&bid=&p={idx}'
    
    return URL
    

