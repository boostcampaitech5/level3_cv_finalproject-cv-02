import pandas as pd
import yaml

from pymongo import MongoClient, UpdateOne
from tqdm import tqdm

class MetaClient:
    
    def __init__(self, collection_name: str, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.DB_NAME = self.conf["mongodb"]["name"]
        self.URL = self.conf["mongodb"]["url"]
        
        self.collection_name = collection_name
        self.client = MongoClient(self.URL, connect=False)
        self.db = self.client[self.DB_NAME]
        self.collection = self.db[collection_name]

    
    def insert(self, df: pd.DataFrame):
        json_list = dataframe2list(df)
        # index 설정해서 중복 제거
        self.collection.create_index("key", unique=True)
        result = self.collection.bulk_write([
            UpdateOne({"key": json["key"]}, {"$set": json}, upsert=True)
                for json in json_list
        ])
        modified_cnt = result.matched_count
        inserted_cnt = len(json_list) - modified_cnt

        return inserted_cnt, modified_cnt

    
    def num_total_items(self):
        return self.collection.count_documents({})


def dataframe2list(df: pd.DataFrame):
    json_list = []
    for _, row in df.iterrows():
        json_template = {
            **row,
        }
        json_list.append(json_template)
    return json_list
