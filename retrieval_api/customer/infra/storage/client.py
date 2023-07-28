import os
import io
import yaml
from google.cloud import storage
from google.oauth2 import service_account


class LocalStorageClient:
    
    def __init__(self, sub_dir: str, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)
        self.SAVE_DIR = os.path.join(self.conf["storage"], sub_dir)

    
    def save(self, filename: str, data: bytes):
        with open(os.path.join(self.SAVE_DIR, filename), "wb") as fp:
            fp.write(data) # -- 서버 로컬스토리지에 이미지 저장


class GCStorageClient:
    
    def __init__(self, sub_dir: str, config_path: str):
        with open(config_path) as f:
            self.conf = yaml.safe_load(f)

        credentials = service_account.Credentials.from_service_account_file(self.conf["key_path"])
        # 구글 스토리지 클라이언트 객체 생성
        self.client = storage.Client(credentials=credentials, project=credentials.project_id)
        self.bucket = self.client.get_bucket(self.conf["bucket_name"])

        self.SAVE_DIR = sub_dir

    
    def save(self, filename: str, data: bytes):
        blob_name = os.path.join(self.SAVE_DIR, filename)
        blob = self.bucket.blob(blob_name)
        blob.upload_from_file(io.BytesIO(data))


    def get(self, filename) -> bytes:
        blob_name = os.path.join(self.SAVE_DIR, filename)
        blob = self.bucket.blob(blob_name)
        data = blob.download_as_bytes()
        return data