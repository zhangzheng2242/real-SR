import time
import numpy as np
import torch.nn as nn
from models.models import Generator
import torch

#加载网络权重
#RealESRGAN_x4plus_anime_6B
model_path = 'weights/vapsr-x2/vapsr-x2net.pth'
#model = Generator(scale=4)

#model.load_state_dict(torch.load(model_path))
#model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
model = torch.load(model_path).cuda()
#剪枝前的网络结构
print(model)

##剪枝
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner   
from nni.compression.pytorch.speedup import ModelSpeedup

config_list = [{ 'sparsity': 0.7, 'op_types': ['Conv2d'] },
{'exclude': True,'op_names': ['upsampler.0','upsampler.2']}            #固定最后一层
]  #'sparsity': 0.7，减掉70%


pruner = L1NormPruner(model, config_list)
_, masks = pruner.compress()
#得到剪枝的mask
pruner._unwrap_model()

# 根据mask，实现网络重构-实现加速
ModelSpeedup(model, dummy_input = torch.ones(1,3,50,50).cuda() , masks_file=masks).speedup_model()    #dummy_input 设置大了运行会炸显存，试验几次该维度大小不影响剪枝结果

#保存剪枝后的网络模型（需要保存网络的结构与权重）
save_path ='weights/vapsr-x2/vapsr-x2net-70%-yuan.pth'
torch.save(model, save_path) 
print("="*10)

#剪枝后的网络结构
print(model)

#得到的模型权重直接加载再训练，学习率稍微小一点（微调）










