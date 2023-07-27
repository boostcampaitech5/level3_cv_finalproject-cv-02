# streamlit
import streamlit as st

# external-library
from PIL import Image
import io

# built-in library
import requests
import base64

# custom-library
from utils.open import open_yaml


class Meotandard:
    """모든 모델의 부모 클래스입니다.
    MeotandardSeg, MeotandardRetrieval, MeotandardViton은 이를 상속받아 사용합니다.
    각 모델에서 공통적으로 사용하는 값들을 여기에 정의해주시면 됩니다.
    """
    def __init__(self):
        # 각 모델 별 ip 주소를 정의합니다.
        api_address = open_yaml("./api_address.yaml")

        self.seg_api_address = api_address["segmentation"]
        self.retrieval_api_address = api_address["retrieval"]
        self.viton_api_adrress = api_address["viton"]


class MeotandardSeg(Meotandard):
    def __init__(self):
        super(MeotandardSeg, self).__init__()
    
    @st.cache_data
    def querying_seg_api(_self, img_bytes: bytes, coordinates: dict) -> tuple:
        """img_bytes, coordinates를 기반으로 segmentation api를 호출할 때 사용하는 함수입니다.
        st.cache_data를 이용하여 동일한 img_bytes와 coordinates 값이 들어오면 cache에 저장해둔
        값을 그대로 return함으로써 속도를 향상시켰습니다.

        Args:
            img_bytes (bytes): bytes 객체입니다.
            coordinates (dict): segmentation 시 필요한 x, y 좌표가 담겨 있습니다.

        Returns:
            tuple: (PIL.Image로 변환된 segmented_imgs, bytes 객체로 변환된 segmented_byte_imgs)
        """

        # segmentation api 호출
        files = {"img" : img_bytes.getvalue()}
        response = requests.post(f"{_self.seg_api_address}/seg?x={coordinates['x']}&y={coordinates['y']}", files=files)

        # 응답으로부터 segmentation된 데이터를 가져와 이미지로 변환
        bytes_list = response.json()
        segmented_imgs, seg_and_pad_imgs, segmented_byte_imgs = [], [], []
        for bytes_data in bytes_list:
            img_data = base64.b64decode(bytes_data)
            segmented_byte_imgs.append(img_data)

            img = Image.open(io.BytesIO(img_data))
            segmented_imgs.append(img)
            
            # streamlit-image-select에서 segmente된 이미지 전체를 볼 수 있도록 패딩
            img_w, img_h = img.size
            if img_w > img_h: 
                new_img = Image.new(img.mode, (img_w, img_w), (0, 0, 0))
                new_img.paste(img, (0, int((img_w - img_h) / 2)))
            else :
                new_img = Image.new(img.mode, (img_h, img_h), (0, 0, 0))
                new_img.paste(img, (int((img_h - img_w)/2), 0))
            
            seg_and_pad_imgs.append(new_img)
        
        return (segmented_imgs, seg_and_pad_imgs, segmented_byte_imgs)


class MeotandardRetrieval(Meotandard):
    def __init__(self):
        super(MeotandardRetrieval, self).__init__()
    
    @st.cache_data
    def querying_searchbyfilter_api(_self, img_bytes: bytes, text: str, threshold: float = 0.0) -> list:
        """img_bytes, text, threshold를 기반으로 image retrieval api를 호출할 때 사용하는 함수입니다.

        Args:
            img_bytes (bytes): segment된 img를 bytes 형식으로 변환한 데이터를 사용합니다.
            text (str): 사용자가 입력한 text입니다.
            threshold (float, optional): 검색 시 사용하는 threshold 값입니다. Defaults to 0.0.

        Returns:
            list: 검색 결과로 나온 상품의 메타데이터들이 담겨있습니다.
        """
        files = {'file': img_bytes}
        form_data = {'thresh': threshold, 'text': text}

        # files와 form_data를 기반으로 proxy/search-by-filter에 검색 요청
        response = requests.post(f"{_self.retrieval_api_address}/proxy/search-by-filter", files=files, data=form_data)

        # 상품목록에 담겨 있는 metadata를 반환
        clothes_metadata = response.json()["상품목록"]

        return clothes_metadata
    
    @st.cache_data
    def querying_searchbyimage_api(self,):
        pass


class MeotandardViton(Meotandard):
    def __init__(self):
        super(MeotandardViton, self).__init__()
        pass