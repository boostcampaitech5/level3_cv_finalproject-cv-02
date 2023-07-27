# built-in library
from dataclasses import dataclass


@dataclass
class SearchedCloth():
    """검색된 상품의 정보를 저장하는 dataclass입니다.
    """
    cloth_metadata: dict

    def __post_init__(self):
        self.purchase_link = self.cloth_metadata['link']
        self.cloth_name = self.cloth_metadata['name']
        self.img_link = self.cloth_metadata['image_link']
        self.category = self.cloth_metadata['category']['type']