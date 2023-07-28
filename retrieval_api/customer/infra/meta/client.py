import yaml

from pymongo import MongoClient

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


    # fine_one의 Execution time: 3.6651980876922607 seconds > 실제 검색 시 8.37s
    # find의 Execution time: 0.20023798942565918 seconds
    # find + ids의 순서로 정렬한 것의 Execution time: 0.20313620567321777 seconds > 실제 검색 시 6.11s
    # ids의 순서는 곧 유사도의 순위 >find_one이 find보다 18배나 느림 하지만 유사도 순으로 정렬되어있음.
    # find 후 정렬을 사용하는 것이 find만 사용하는 것보다 0.0003s 차이지만 find_one보다는 압도적 성능을 지닌다.
    # 빠른 검색이 목적: find / 유사도 정렬이 목적: find_one
    def find(self, ids: list[int]):
        documents = self.collection.find({'key': {'$in': ids}})
        documents = sorted(documents, key=lambda doc: ids.index(doc['key']))
        black_list = ['_id', 'path']
        for doc in documents:
            for key in black_list:
                del doc[key]
        return documents


    def num_total_items(self):
        return self.collection.count_documents({})
