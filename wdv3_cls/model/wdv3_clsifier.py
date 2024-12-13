import torch
import torch.nn as nn
import torchvision.models as models
import timm
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
import copy
from dataclasses import dataclass
from simple_parsing import field, parse_known_args
from pathlib import Path
import json
import pdb

MODEL_REPO_MAP = {
    "vit": "SmilingWolf/wd-vit-tagger-v3",
    "swinv2": "SmilingWolf/wd-swinv2-tagger-v3",
    "convnext": "SmilingWolf/wd-convnext-tagger-v3",
    "vit-local": "/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/config/vit_fuse_lcavg_l.json",
    "vit-local-finetune": "/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/config/vit_finetune_mengbao.json",
}

MODEL_LOCAL_PATH = {
    "vit-local": "/data/liuhaoxiang/pretrained_models/wd-vit-tagger-v3/model_resize.safetensors",
    "vit-local-finetune": "/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/experiments/wdv3_comics_finetune_line/model/best_epoch_model.pth"
}

@dataclass
class ScriptOptions:
    # image_file: Path = field(positional=True)
    model: str = field(default="vit")
    # gen_threshold: float = field(default=0.35)
    # char_threshold: float = field(default=0.75)

@dataclass
class Wdv3ViTModelOptions:
    model: str = field(default="vit")

@dataclass
class ConcatWdv3ViTModelOptions:
    model: str = field(default="vit-local")


@dataclass
class FinetuneWdv3ViTModelOptions:
    model: str = field(default="vit-local-finetune")


class DoubleWdv3Classifier(nn.Module):
    def __init__(self, model_cfg, num_classes=600):
        super().__init__()
        self.repo_id = MODEL_REPO_MAP.get(model_cfg.model)

        self.model_line = timm.create_model("hf-hub:" + self.repo_id)
        self.model_lc_avg = timm.create_model("hf-hub:" + self.repo_id)

        # fmodel = timm.create_model("hf-hub:" + "SmilingWolf/wd-vit-tagger-v3", pretrained=False)
        self.state_dict_pretrain_line = timm.models.load_state_dict_from_hf(self.repo_id)
        self.state_dict_pretrain_lc_avg = copy.deepcopy(self.state_dict_pretrain_line)

        self.model_line.load_state_dict(self.state_dict_pretrain_line)
        self.model_lc_avg.load_state_dict(self.state_dict_pretrain_lc_avg)

        self.num_features = self.model_line.head.in_features

        self.model_line.head = nn.Identity()
        self.model_lc_avg.head = nn.Identity()

        self.model_head = nn.Linear(self.num_features * 2, num_classes)

        # self.model.head = nn.Linear(self.num_features, num_classes)
        print(f"Model head - num_classes: {self.model_head.out_features}")

    def forward(self, x):
        b, c, h, w = x.shape

        assert c % 2 == 0, "Channel inputs is ERROR, C Must Be Even"

        x_line, x_lc_avg = x.split(c // 2, dim=1)
        x_line = self.model_line(x_line)
        x_lc_avg = self.model_lc_avg(x_lc_avg)

        x = torch.cat([x_line, x_lc_avg], dim=1)
        x = self.model_head(x)
        return x
    
    def check_model(self):
        return self.model_line


class Wdv3Classifier(nn.Module):
    def __init__(self, model_cfg, num_classes=600):
        super().__init__()
        self.repo_id = MODEL_REPO_MAP.get(model_cfg.model)
        if 'local' not in model_cfg.model:
            self.model = timm.create_model("hf-hub:" + self.repo_id)
            # fmodel = timm.create_model("hf-hub:" + "SmilingWolf/wd-vit-tagger-v3", pretrained=False)
            self.state_dict_pretrain = timm.models.load_state_dict_from_hf(self.repo_id)

        else:
            with open(self.repo_id, 'r', encoding='utf-8') as file:
                model_config = json.load(file)

            self.model = timm.create_model(
                model_config["architecture"],
                pretrained=False,  
                num_classes=model_config["num_classes"],
                pretrained_cfg=model_config["pretrained_cfg"],
                **model_config["model_args"]  
            )
            local_pretrain_weights_path = MODEL_LOCAL_PATH.get(model_cfg.model)
            
            self.state_dict_pretrain = torch.load(local_pretrain_weights_path)

        self.model.load_state_dict(self.state_dict_pretrain)

        self.num_features = self.model.head.in_features
        self.model.head = nn.Linear(self.num_features, num_classes)
        print(f"Model head - num_classes: {self.model.head.out_features}")

    def forward(self, x):
        return self.model(x)
    
    def check_model(self):
        return self.model

class Wdv3Classifier_downstream(nn.Module):
    def __init__(self, model_cfg, num_classes=4, local_pretrain_model_num_classes=600):
        super().__init__()
        self.repo_id = MODEL_REPO_MAP.get(model_cfg.model)
        if 'local' not in model_cfg.model:
            self.model = timm.create_model("hf-hub:" + self.repo_id)
            # fmodel = timm.create_model("hf-hub:" + "SmilingWolf/wd-vit-tagger-v3", pretrained=False)
            self.state_dict_pretrain = timm.models.load_state_dict_from_hf(self.repo_id)

        else:
            with open(self.repo_id, 'r', encoding='utf-8') as file:
                model_config = json.load(file)

            self.model = timm.create_model(
                model_config["architecture"],
                pretrained=False,  
                num_classes=local_pretrain_model_num_classes,
                pretrained_cfg=model_config["pretrained_cfg"],
                **model_config["model_args"]  
            )
            local_pretrain_weights_path = MODEL_LOCAL_PATH.get(model_cfg.model)
            
            self.state_dict_pretrain = torch.load(local_pretrain_weights_path)
        
        self.load_state_dict(self.state_dict_pretrain['state_dict'])
        self.model.requires_grad_(False)
        self.num_features = self.model.head.in_features
        self.model.head = nn.Linear(self.num_features, num_classes)
        self.model.head.requires_grad_(True)
        # pdb.set_trace()

        print(f"Model head - num_classes: {self.model.head.out_features}")

    def forward(self, x):
        return self.model(x)
    
    def check_model(self):
        return self.model

if __name__ == '__main__':

    # with open("/data/liuhaoxiang/workspace/new_worksp/wdv3-timm-cls/config/vit_fuse_lcavg_l.json", 'r', encoding='utf-8') as file:

    #     model_config = json.load(file)

    # model = timm.create_model(
    #     model_config["architecture"],
    #     pretrained=False,  # 防止从在线源加载预训练权重
    #     num_classes=model_config["num_classes"],
    #     pretrained_cfg=model_config["pretrained_cfg"],
    #     # **model_config["model_args"]  # 传递额外的模型参数
    # )
    # state_dict_pretrain = timm.models.load_state_dict('/data/liuhaoxiang/pretrained_models/wd-vit-tagger-v3/model.safetensors')

    # for key in state_dict_pretrain.keys():
    #     if "patch_embed.proj.weight" in key:
    #         # 获取原始权重
    #         original_weight = state_dict_pretrain[key]

    #         shape_cur = model.patch_embed.proj.weight.shape
    #         # 创建新的权重，形状为 [6, 16, 16]
    #         new_weight = torch.zeros(shape_cur, device=original_weight.device)
            
    #         # 复制原始权重到新的权重
    #         new_weight[:, :3, :, :] = original_weight
            
    #         # 更新状态字典
    #         state_dict_pretrain[key] = new_weight

    # torch.save(state_dict_pretrain, '/data/liuhaoxiang/pretrained_models/wd-vit-tagger-v3/model_resize.safetensors')
    # pdb.set_trace()

    tensor = torch.randn(1, 6, 448, 448)
    wdv3_classifier_2_stream = DoubleWdv3Classifier(Wdv3ViTModelOptions)



    # wdv3_classifier = Wdv3Classifier(ConcatWdv3ViTModelOptions)
    # model_test = wdv3_classifier.check_model()
    pdb.set_trace()
    # print(model_test.parameters)