#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Author  :   Peike Li
@Contact :   peike.li@yahoo.com
@File    :   simple_extractor.py
@Time    :   8/30/19 8:59 PM
@Desc    :   Simple Extractor
@License :   This source code is licensed under the license found in the
             LICENSE file in the root directory of this source tree.
"""

# torch
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# external-library
import numpy as np
from PIL import Image
import warnings

# built-in library
import os
import os.path as osp
from tqdm import tqdm
from pathlib import Path

# custom-library
import infra.preprocess.human_parser.schp.networks as networks
from infra.preprocess.human_parser.schp.utils.transforms import transform_logits
from infra.preprocess.human_parser.schp.datasets.simple_extractor_dataset import SimpleFolderTestDataset
from collections import OrderedDict

warnings.filterwarnings('ignore')


class SCHP():
    def __init__(self, config):
        super(SCHP, self).__init__()
        self.dataset_settings = {
            'lip': {
                'input_size': [473, 473],
                'num_classes': 20,
                'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                        'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                        'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
            },
            'atr': {
                'input_size': [512, 512],
                'num_classes': 18,
                'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                        'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
            },
            'pascal': {
                'input_size': [512, 512],
                'num_classes': 7,
                'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
            }
        }
        
        # config 설정
        self.args = config

        # 현재 이 .py가 있는 부모 경로를 절대경로로 지정(./human_parser 폴더 경로)
        PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()

        # pretrained 모델 경로 
        self.resnet_path = osp.join(PROJECT_ROOT, 'pretrained', 'resnet101-imagenet.pth')
        self.human_parsing_path = osp.join(PROJECT_ROOT, 'pretrained', 'final.pth')

        
    def get_palette(self, num_cls):
        """ Returns the color map for visualizing the segmentation mask.
        Args:
            num_cls: Number of classes
        Returns:
            The color map
        """
        n = num_cls
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette


    def inference(self, storage_root: str, img_name: str):
        args = self.args

        device = [int(i) for i in args.human_gpu.split(',')]
        assert len(device) == 1

        if not args.human_gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.human_gpu

        num_classes = self.dataset_settings[args.human_dataset]['num_classes']
        input_size = self.dataset_settings[args.human_dataset]['input_size']
        # label = self.dataset_settings[args.human_dataset]['label']
        # print("Evaluating total class number {} with {}".format(num_classes, label))

        model = networks.init_model('resnet101', num_classes=num_classes, pretrained=self.resnet_path)

        state_dict = torch.load(self.human_parsing_path)['state_dict']

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        model.cuda()
        model.eval()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
        ])

        img_path = osp.join(storage_root, "raw_data/person", img_name)
        save_path = osp.join(storage_root, "preprocess/human_parse", img_name)

        dataset = SimpleFolderTestDataset(img_path, input_size=input_size, transform=transform)
        dataloader = DataLoader(dataset)

        palette = self.get_palette(num_classes)

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                img, meta = batch
                img_name = meta['name'][0]
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output = model(img.to(device[0]))
                upsample = torch.nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
                upsample_output = upsample(output[0][-1][0].unsqueeze(0))
                upsample_output = upsample_output.squeeze()
                upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

                logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=input_size)
                parsing_result = np.argmax(logits_result, axis=2)
                output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
                output_img.putpalette(palette)
                output_img.save(save_path, "PNG")

                # check
                save_state = False
                if osp.exists(save_path) and osp.getsize(save_path):
                    save_state = True
                
                if args.logits:
                    logits_result_path = os.path.join(self.output_path, img_name[:-4] + '.npy')
                    np.save(logits_result_path, logits_result)
        
        return save_state
