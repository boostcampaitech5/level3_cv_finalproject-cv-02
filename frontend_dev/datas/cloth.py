# built-in library
from dataclasses import dataclass


@dataclass
class SearchedCloth():
    """검색된 상품의 정보를 저장하는 dataclass입니다.
    """
    cloth_metadata: dict

    def __post_init__(self):
        # 이미지와 쇼핑몰 url
        self.img_link = self.cloth_metadata['image_link']
        self.purchase_link = self.cloth_metadata['link']

        # 상품 정보
        self.cloth_name = self.cloth_metadata['name']
        self.price = self.cloth_metadata['price']

        # 상품 메타데이터
        self.tag = self.cloth_metadata['meta']
        self.category = self.cloth_metadata['category']

        # 생성 가능 여부 판단
        self.can_tryon = self.cloth_metadata['check']

