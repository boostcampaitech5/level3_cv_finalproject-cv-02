import json
import os
import warnings
from pathlib import Path
import sys
from typing import Dict
import cv2

import numpy as np
from numpy.linalg import lstsq
from PIL import Image, ImageDraw, ImageOps


import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from accelerate import Accelerator
from diffusers import DDIMScheduler

from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from infra.viton.ladi_viton.src.models.AutoencoderKL import AutoencoderKL
from infra.viton.ladi_viton.src.utils.encode_text_word_embedding import encode_text_word_embedding
from infra.viton.ladi_viton.src.utils.set_seeds import set_seed
from infra.viton.ladi_viton.src.vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

from infra.viton.ladi_viton.src.utils.labelmap import LIP_map 
from infra.viton.ladi_viton.src.utils.posemap import kpoint_to_heatmap
from infra.viton.ladi_viton.src.utils.posemap import get_coco_body25_mapping

warnings.filterwarnings('ignore') # 짜잘한 에러 무시
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

label_map = LIP_map

class LadiVton():
    def __init__(self):
        super(LadiVton, self).__init__()
        
        # base setting
        self.args = {
            'output_dir': '../result/',
            'pretrained_model_name_or_path' : "stabilityai/stable-diffusion-2-inpainting",
            'mixed_precision' : 'no',
            'enable_xformers_memory_efficient_attention' : True,
            'size' : (512, 384),
            'radius' : 5,
            'seed' : 20,
            "num_vstar" : 16,
        }


        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        

        # Setup accelerator and device.
        self.accelerator = Accelerator(mixed_precision=self.args['mixed_precision'])
        self.device = self.accelerator.device

        # create an model
        self.upper_model = self.create_model()
        self.lower_model = self.create_model("lower_body")


    def data_preprocessing(self, data_name:Dict):
        outputlist = ['image', 'pose_map', 'inpaint_mask', 'im_mask', 'category',  'cloth']
        
        category = data_name.get("category")
        # 옷 전처리
        cloth = Image.open(data_name.get("cloth"))
        cloth_mask = Image.open(data_name.get("cloth_mask"))

        ## 배경 제거
        cloth = Image.composite(ImageOps.invert(cloth_mask.convert('L')), cloth, ImageOps.invert(cloth_mask.convert('L')))
        cloth = cloth.resize((self.args['size'][1], self.args['size'][0]))
        cloth = self.transform(cloth) # [-1, 1]

        # im 전처리
        image = Image.open(data_name.get("im")).convert('RGB')
        image_mask = Image.open(data_name.get("image_mask"))

        ## 배경 제거
        image = Image.composite(ImageOps.invert(image_mask.convert('L')), image, ImageOps.invert(image_mask.convert('L')))
        image = image.resize((self.args['size'][1], self.args['size'][0]))
        image = self.transform(image) # [-1, 1]

        # posemap
        im_parse = Image.open(data_name.get("im_parse"))
        im_parse = im_parse.resize((self.args['size'][1], self.args['size'][0]), Image.NEAREST)
        parse_array = np.array(im_parse)

        parse_shape = (parse_array > 0).astype(np.float32) #not background area

        # 머리 부분
        parse_head = (parse_array == label_map['hat']).astype(np.float32) + \
                     (parse_array == label_map['hair']).astype(np.float32) + \
                     (parse_array == label_map['sunglasses']).astype(np.float32) + \
                     (parse_array == label_map['face']).astype(np.float32)

        # 고정 부분
        parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                            (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                            (parse_array == label_map["hat"]).astype(np.float32) + \
                            (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                            (parse_array == label_map["scarf"]).astype(np.float32) + \
                            (parse_array == label_map["glove"]).astype(np.float32)

        # 변화 가능 부분
        parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)
                    
        arms = (parse_array == 14).astype(np.float32) + (parse_array == 15).astype(np.float32)

        # 카테고리 기준으로 갱신
        if data_name.get("category") == 'upper_body':
            parse_cloth = (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                          (parse_array == label_map["coat"]).astype(np.float32)

            parse_mask = (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                         (parse_array == label_map["coat"]).astype(np.float32)

            parser_mask_fixed += (parse_array == label_map['skirt']).astype(np.float32) + \
                                 (parse_array == label_map['pants']).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        elif data_name.get("category") == 'lower_body':
            parse_cloth = (parse_array == label_map['pants']).astype(np.float32)
            parse_mask = (parse_array == label_map['pants']).astype(np.float32) + \
                         (parse_array == label_map['left_leg']).astype(np.float32) + \
                         (parse_array == label_map['right_leg']).astype(np.float32)
            
            parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) +\
                                 (parse_array == label_map["coat"]).astype(np.float32) + \
                                 (parse_array == label_map["left_arm"]).astype(np.float32) + \
                                 (parse_array == label_map["right_arm"]).astype(np.float32)

            parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

        else:
            raise NotImplementedError

        parse_head = torch.from_numpy(parse_head)  # [0,1]
        parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
        parse_mask = torch.from_numpy(parse_mask)  # [0,1]
        parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
        parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

        parse_without_cloth = np.logical_and(parse_shape, np.logical_not(parse_mask))
        parse_mask = parse_mask.cpu().numpy()

        # Shape
        parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
        parse_shape = parse_shape.resize((self.args['size'][1] // 16,  self.args['size'][0] // 16), Image.BILINEAR)
        parse_shape = parse_shape.resize((self.args['size'][1], self.args['size'][0]), Image.BILINEAR)
        shape = self.transform2D(parse_shape)

        # Load pose points
        with open(data_name.get("pose"), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]


        pose_mapping = get_coco_body25_mapping()

        point_num = len(pose_mapping)

        pose_map = torch.zeros(point_num, self.args['size'][0], self.args['size'][1])
        r = self.args['radius'] *(self.args['size'][0]/512.0)
        im_pose = Image.new('L', (self.args['size'][1], self.args['size'][0]))
        pose_draw = ImageDraw.Draw(im_pose)
        neck = Image.new('L', (self.args['size'][1], self.args['size'][0]))
        neck_draw = ImageDraw.Draw(neck)

        for i in range(point_num):
            one_map = Image.new('L', (self.args['size'][1], self.args['size'][0]))
            draw = ImageDraw.Draw(one_map)

            point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
            point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

            if point_x > 1 and point_y > 1:
                draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                if i == 2 or i == 5:
                    neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                    'white')
            one_map = self.transform2D(one_map)
            pose_map[i] = one_map[0]


        d = []
        for idx in range(point_num):
            ux = pose_data[pose_mapping[idx], 0]  # / (192)
            uy = (pose_data[pose_mapping[idx], 1])  # / (256)

            # sclae posemap points
            px = ux 
            py = uy 

            d.append(kpoint_to_heatmap(np.array([px, py]), (self.args['size'][0], self.args['size'][1]), 9))
        
        pose_map = torch.stack(d)


        # just for visualization
        im_pose = self.transform2D(im_pose)
        

        im_arms = Image.new('L', (self.args['size'][1], self.args['size'][0]))
        arms_draw = ImageDraw.Draw(im_arms)

        if data_name.get("category") == 'upper_body' or data_name.get("category") == 'lower_body':
            with open(data_name.get("pose"), 'r') as f:
                data = json.load(f)
                data = data['people'][0]['pose_keypoints_2d']
                data = np.array(data)
                data = data.reshape((-1, 3))[:, :2]

                # data[:, 0] = data[:, 0] * (self.args['size'][1] / 768)
                # data[:, 1] = data[:, 1] * (self.args['size'][0] / 1024)

                shoulder_right = tuple(data[pose_mapping[2]])
                shoulder_left = tuple(data[pose_mapping[5]])
                elbow_right = tuple(data[pose_mapping[3]])
                elbow_left = tuple(data[pose_mapping[6]])
                wrist_right = tuple(data[pose_mapping[4]])
                wrist_left = tuple(data[pose_mapping[7]])

                ARM_LINE_WIDTH = int(90 / 512 * self.args['size'][0])
                if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                    if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                    if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                        arms_draw.line(
                            np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                else:
                    arms_draw.line(np.concatenate(
                        (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                        np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            
                hands = np.logical_and(np.logical_not(im_arms), arms)

                if data_name.get("category") == 'upper_body':
                    parse_mask += im_arms
                    parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)

            parser_mask_fixed = np.logical_or(parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))


            #Image.fromarray(np.array(parse_mask, dtype=np.uint8)*255).convert('RGB').save('../input_checks/parse_mask1.png')

            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = cv2.dilate(parse_mask, np.ones((7,7), np.uint16), iterations=5)

            parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            visual_inpaint_mask = inpaint_mask

            inpaint_mask = inpaint_mask.unsqueeze(0)
            parse_mask_total = parse_mask_total.numpy()
            parse_mask_total = parse_array * parse_mask_total
            parse_mask_total = torch.from_numpy(parse_mask_total)
       
        # inpaint랑 im_mask 시각화 -> for check
        #Image.fromarray(np.array(visual_inpaint_mask, dtype=np.uint8)*255).convert('RGB').save('../input_checks/inpaint_mask.png')
        #save_image(im_mask, '../input_checks/im_mask.png')

        result = {}
        for k in outputlist:
            result[k] = vars()[k]

        return result
    
    # TODO: 동시 요청이 들어왔을 때 batch 단위로 처리하는 방법 찾아보기
    def create_model(self, category="upper_body"):
        """ladi_vton pipeline을 생성할 때 사용하는 함수입니다.
        """
        # Load scheduler, tokenizer and models.
        val_scheduler = DDIMScheduler.from_pretrained(self.args['pretrained_model_name_or_path'], subfolder="scheduler")
        val_scheduler.set_timesteps(50, device=self.device)
        text_encoder = CLIPTextModel.from_pretrained(self.args['pretrained_model_name_or_path'], subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(self.args['pretrained_model_name_or_path'], subfolder="vae")
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        tokenizer = CLIPTokenizer.from_pretrained(self.args['pretrained_model_name_or_path'], subfolder="tokenizer")
        
        
        PRETRAINED = {'upper_body':'vitonhd', 'lower_body':'dresscode'} #카테고리에 따라 가중치 설정
        unet = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='extended_unet', dataset=PRETRAINED[f'{category}'])
        emasc = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='emasc', dataset=PRETRAINED[f'{category}'])
        inversion_adapter = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='inversion_adapter', dataset=PRETRAINED[f'{category}'])
        tps, refinement = torch.hub.load(repo_or_dir='miccunifi/ladi-vton', source='github', model='warping_module', dataset=PRETRAINED[f'{category}'])

        int_layers = [1, 2, 3, 4, 5]

        # Enable xformers memory efficient attention if requested
        if self.args['enable_xformers_memory_efficient_attention']:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # cast to weight_dtype 
        self.weight_dtype = torch.float32

        text_encoder.to(self.device, dtype=self.weight_dtype)
        vae.to(self.device, dtype=self.weight_dtype)
        emasc.to(self.device, dtype=self.weight_dtype)
        inversion_adapter.to(self.device, dtype=self.weight_dtype)
        unet.to(self.device, dtype=self.weight_dtype)
        tps.to(self.device, dtype=self.weight_dtype)
        refinement.to(self.device, dtype=self.weight_dtype)
        vision_encoder.to(self.device, dtype=self.weight_dtype)

        # set to eval
        text_encoder.eval()
        vae.eval()
        emasc.eval()
        inversion_adapter.eval()
        unet.eval()
        tps.eval()
        refinement.eval()
        vision_encoder.eval()

        # Create the pipeline
        val_pipe = StableDiffusionTryOnePipeline(
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=val_scheduler,
            emasc=emasc,
            emasc_int_layers=int_layers,
        ).to(self.device)

        models = {'text_encoder': text_encoder, 'vision_encoder':vision_encoder,
                  'processor': processor, 'tokenizer': tokenizer,
                  'inversion_adapter': inversion_adapter,'tps': tps,
                  'refinement': refinement, 'val_pipe':val_pipe}
        
        return models
        
        
    def inference(self, storage_root: str, p_img_name: str, c_img_name: str, category: str):
        inputlist = ['im', 'image_mask', 'cloth', 'cloth_mask', 'im_parse', 'pose', 'category']
        category = category

        im = os.path.join(storage_root, 'raw_data/person', p_img_name)
        image_mask = os.path.join(storage_root, 'preprocess/mask/person', p_img_name)
        
        cloth  = os.path.join(storage_root, 'raw_data/cloth', c_img_name)
        cloth_mask = os.path.join(storage_root, 'preprocess/mask/cloth', c_img_name)

        im_parse = os.path.join(storage_root, 'preprocess/human_parse', p_img_name.replace('.png', '.png'))
        pose = os.path.join(storage_root, 'preprocess/pose/keypoints', p_img_name.replace('.png', '.json'))
        

        # 전처리 input dict 생성
        data_name = {}
        for k in inputlist:
            data_name[k] = vars()[k]

        # 전처리한 데이터 
        batch = self.data_preprocessing(data_name=data_name)
        
        # category에 따라 사용할 모델을 지정
        if category == "upper_body":
            model = self.upper_model
            print("upper model이 선택되었습니다.")
        elif category == "lower_body":
            model = self.lower_model
            print("lower model이 선택되었습니다.")
        else:
            print("Wrong category: category is one of the [upper_body, lower_body]")
            raise

        text_encoder, vision_encoder = model['text_encoder'], model['vision_encoder'], 
        processor, tokenizer = model['processor'], model['tokenizer']
        inversion_adapter, tps = model['inversion_adapter'], model['tps']
        refinement, val_pipe = model['refinement'], model['val_pipe']

        # 난수 생성
        generator = torch.Generator("cuda").manual_seed(self.args['seed'])

        # 전처리된 데이터 받아오고 weight_dtype에 맞춰 주기
        model_img = batch.get("image").unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        mask_img = batch.get("inpaint_mask").unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        if mask_img is not None:
            mask_img = mask_img.to(self.device, dtype=self.weight_dtype)
        pose_map = batch.get("pose_map").unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        category = [batch.get("category")]
        cloth = batch.get("cloth").unsqueeze(0).to(self.device, dtype=self.weight_dtype)
        im_mask = batch.get('im_mask').unsqueeze(0).to(self.device, dtype=self.weight_dtype)

        # Generate the warped cloth (와핑)
        low_cloth = torchvision.transforms.functional.resize(cloth, (256, 192),
                                                             torchvision.transforms.InterpolationMode.BILINEAR,
                                                             antialias=True)
        low_im_mask = torchvision.transforms.functional.resize(im_mask, (256, 192),
                                                               torchvision.transforms.InterpolationMode.BILINEAR,
                                                               antialias=True)
        low_pose_map = torchvision.transforms.functional.resize(pose_map, (256, 192),
                                                                torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True)
        agnostic = torch.cat([low_im_mask, low_pose_map], 1)
        low_grid, theta, rx, ry, cx, cy, rg, cg = tps(low_cloth, agnostic)

        # We upsample the grid to the original image size and warp the cloth using the predicted TPS parameters
        highres_grid = torchvision.transforms.functional.resize(low_grid.permute(0, 3, 1, 2),
                                                                size=(512, 384),
                                                                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                                                                antialias=True).permute(0, 2, 3, 1)

        warped_cloth = F.grid_sample(cloth, highres_grid, padding_mode='border')

        # Refine the warped cloth using the refinement network
        warped_cloth = torch.cat([im_mask, pose_map, warped_cloth], 1)
        warped_cloth = refinement(warped_cloth)
        warped_cloth = warped_cloth.clamp(-1, 1)

        # Get the visual features of the in-shop cloths
        input_image = torchvision.transforms.functional.resize((cloth + 1) / 2, (224, 224),
                                                               antialias=True).clamp(0, 1)
        processed_images = processor(images=input_image, return_tensors="pt")
        clip_cloth_features = vision_encoder(
            processed_images.pixel_values.to(model_img.device, dtype=self.weight_dtype)).last_hidden_state

        ## TPS - 와핑 끝

        # Compute the predicted PTEs
        word_embeddings = inversion_adapter(clip_cloth_features.to(model_img.device))
        word_embeddings = word_embeddings.reshape((word_embeddings.shape[0], self.args['num_vstar'], -1))

        category_text = {
            'upper_body': 'an upper body garment',
            'lower_body': 'a lower body garment',
        }
        num_vstar = self.args['num_vstar']
        text = [f'a photo of a model wearing {category_text[category]} {" $ " * num_vstar}' for category in [batch['category']]]

       # Tokenize text
        tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                   truncation=True, return_tensors="pt").input_ids
        tokenized_text = tokenized_text.to(word_embeddings.device)

        # Encode the text using the PTEs extracted from the in-shop cloths
        encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                           word_embeddings, self.args['num_vstar']).last_hidden_state

        # Generate images
        generated_images = val_pipe(
            image=model_img,
            mask_image=mask_img,
            pose_map=pose_map,
            warped_cloth=warped_cloth,
            prompt_embeds=encoder_hidden_states,
            height=512,
            width=384,
            guidance_scale=7.5,
            num_images_per_prompt=1,
            generator=generator,
            cloth_input_type='warped',
            num_inference_steps=50
        ).images


        # Save image
        self.args['output_dir'] = os.path.join(storage_root, 'viton')
        os.makedirs(os.path.join(self.args['output_dir']), exist_ok=True)
        
        # Origin_image read
        origin_image = Image.open(im).convert('RGB')
        origin_size = origin_image.size

        # Input size로 이미지 변환 (원본 사이즈로)
        p_img_name = p_img_name.replace(".jpg", ".png")
                
        image = generated_images[0]
        image = image.resize(origin_size)
        image = image.convert("RGB")

        image.save(os.path.join(self.args['output_dir'], p_img_name), format="PNG")


        # # Free up memory
        # del val_pipe
        # del text_encoder
        # del vae
        # del emasc
        # del unet
        # del tps
        # del refinement
        # del vision_encoder
        # torch.cuda.empty_cache()

        return image # PIL.Image


## TEST 
if __name__ == "__main__":
    model = LadiVton()
    storage_root = '/opt/ml/VIT-ON-Demo/backend/infra/viton/ladi-viton/src/test_data'
    img_name = '4.jpg'
    category = 'upper_body'

    model.inference(storage_root, img_name, category)
