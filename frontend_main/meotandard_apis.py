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
    
    @st.cache_data(show_spinner="검색에 필요한 상품 정보를 찾는 중입니다...")
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
            
            # streamlit-image-select에서 segment된 이미지 전체를 볼 수 있도록 패딩
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
    
    @st.cache_data(show_spinner="이미지와 텍스트를 활용하여 상품 정보를 검색중입니다...")
    def querying_searchbyfilter_api(_self, img_bytes: bytes, text: str, threshold: float = 0.0) -> list:
        """img_bytes, text, threshold를 기반으로 image retrieval api를 호출할 때 사용하는 함수입니다.

        Args:
            img_bytes (bytes): segment된 img를 bytes 형식으로 변환한 데이터를 사용합니다.
            text (str): 사용자가 입력한 text입니다.
            threshold (float, optional): 검색 시 사용하는 threshold 값입니다. Defaults to 0.0.

        Returns:
            list: 검색 결과로 나온 상품의 메타 데이터들이 담겨있습니다.
        """
        files = {'file': img_bytes}
        form_data = {'thresh': threshold, 'text': text}

        # files와 form_data를 기반으로 proxy/search-by-filter에 검색 요청
        response = requests.post(f"{_self.retrieval_api_address}/proxy/search-by-filter", files=files, data=form_data)

        translation_result = response.json()["번역결과"]
        clothes_metadata = response.json()["상품목록"]

        return translation_result, clothes_metadata
    
    @st.cache_data(show_spinner="이미지를 활용하여 상품 정보를 검색중입니다...")
    def querying_searchbyimage_api(_self, img_bytes: bytes, threshold: float = 0.0):
        files = {'file': img_bytes}
        form_data = {'thresh': threshold}

        response = requests.post(f"{_self.retrieval_api_address}/proxy/search-by-image", files=files, data=form_data)

        clothes_metadata = response.json()["상품목록"]

        return clothes_metadata

    @st.cache_data(show_spinner="텍스트를 활용하여 상품 정보를 검색중입니다...")
    def querying_searchbytext_api(_self, text: str, threshold: float = 0.0):
        form_data = {'thresh': threshold, 'text': text}

        response = requests.post(f"{_self.retrieval_api_address}/proxy/search-by-text", data=form_data)

        translation_result = response.json()["번역결과"]
        clothes_metadata = response.json()["상품목록"]

        return (translation_result, clothes_metadata)


class MeotandardViton(Meotandard):
    def __init__(self):
        super(MeotandardViton, self).__init__()

    @st.cache_data(show_spinner="이미지를 저장 중입니다...")
    def querying_saveimg_api(_self, img_bytes: bytes, mode: str = 'cloth') -> tuple:
        """img_bytes를 기반으로 생성 서버의 로컬 스토리지에 이미지 저장을 요청합니다.

        Args:
            img_bytes (bytes): bytes 형식의 사람 혹은 상품 데이터입니다.

        Returns:
            tuple: 데이터가 올바르게 저장되었는지를 나타내는 상태와 이미지의 이름을 반환합니다.
        """
        files = {"img": img_bytes}
        data = {"mode": mode}
        response = requests.post(url=f"{_self.viton_api_adrress}/proxy/save", files=files, data=data)
        
        save_state, im_name = response.json()['save_state'], response.json()['im_name']

        return (save_state, im_name)

    @st.cache_data(show_spinner="의류 상품 이미지를 처리 중입니다...")
    def querying_ppcloth_api(_self, img_name: str) -> str:
        """생성 서버의 스토리지에 저장된 상품 이미지의 이름을 바탕으로 masking을 요청합니다.

        Args:
            img_name (str): 생성 서버의 스토리지에 저장된 상품 이미지의 이름입니다.

        Returns:
            str: 데이터가 올바르게 저장되었는지를 나타내는 상태 반환
        """
        data = {"img_name": img_name}
        response = requests.post(url=f"{_self.viton_api_adrress}/proxy/preprocess/cloth", data=data)

        return response.text
    
    @st.cache_data(show_spinner="가상 피팅에 활용할 정보들을 처리 중입니다...")
    def querying_ppperson_api(_self, img_name: str, model_name: str = 'ladi_viton') -> dict:
        """생성 서버의 스토리지에 저장된 사람 이미지의 이름을 바탕으로, 생성에 필요한
        모든 전처리를 요청합니다.

        Args:
            img_name (str): 생성 서버의 스토리지에 저장된 사람 이미지의 이름입니다.
            model_name (str, optional): 생성 모델마다 사용하는 전처리가 다르기 떄문에, 이름을 사용하여 요청합니다.
            230723 기준 hr-viton, ladi-viton 모델을 지원합니다. Defaults to 'ladi-viton'.

        Returns:
            dict: 모든 전처리 과정의 상태 정보가 저장되어있습니다.
        """
        data = {"img_name": img_name, "model_name": model_name}
        response = requests.post(url=f"{_self.viton_api_adrress}/proxy/preprocess/person", data=data)
        
        return response.json()
    
    @st.cache_data(show_spinner="가상 피팅을 진행중입니다, 잠시만 기다려주세요...")
    def querying_genvitonimg_api(_self, p_img_name: str, c_img_name: str, model_name: str = 'ladi_viton', category: str = 'upper_body'):
        """사람 이미지와 상품 이미지, 생성 모델의 이름을 가지고 viton api를 요청합니다.

        Args:
            p_img_name (str): 생성 서버의 스토리지에 저장된 사람 이미지의 이름입니다.
            c_img_name (str): 생성 서버의 스토리지에 저장된 의류 이미지의 이름입니다.
            model_name (str, optional): 생성 모델의 이름입니다.
            230723 기준 hr-viton, ladi-viton 모델을 지원합니다. Defaults to 'ladi-viton'.
        """
        data = {"p_img_name": p_img_name, "c_img_name": c_img_name,
                "model_name": model_name, "category": category}
        
        response = requests.post(url=f"{_self.viton_api_adrress}/proxy/generate", data=data)

        # base64 데이터를 image로 변환
        tryon_img = Image.open(io.BytesIO(base64.b64decode(response.content)))

        return tryon_img