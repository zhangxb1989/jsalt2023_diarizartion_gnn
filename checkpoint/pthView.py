import torch

# 加载.pth 文件
model_state_dict = torch.load('deepglint_sampler.pth')

# 查看模型状态字典的键和值
for key, value in model_state_dict.items():
    print(f'Key: {key}')
    print(f'Value: {value}')
    print('---')