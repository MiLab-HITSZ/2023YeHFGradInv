# Copyright (C) Machine Intelligence Laboratory, Harbin Institute of Technology, Shenzhen
# All rights reserved
# @Time        : 2023/7/18 1:35 下午
# @Author      : Zipeng Ye
# @Affiliation : Harbin Institute of Technology, Shenzhen
# @File        : get_plot_cv.py

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import breaching
from custom_dataset import CustomData


if __name__ == "__main__":
    model_name = 'resnet34'
    pretrained = True
    model = getattr(torchvision.models, model_name)(pretrained=pretrained)
    model.eval()
    cus_data = CustomData(data_dir='custom_data/web_image_224/', dataset_name='ImageNet', number_data_points=40)
    inputs = cus_data.process_data()['inputs']

    feature = []


    def handle_hook():
        def hook(model, input, output):
            feature.append(input[0].detach().cpu())

        return hook


    for name, module in model.named_modules():
        if name == 'fc':
            module.register_forward_hook(handle_hook())

    out = model(inputs)
    sort = torch.sort(feature[0], dim=1)[0]
    std = torch.std(sort, dim=0)
    mean = torch.mean(sort, dim=0)
    cv = (std / mean).reshape(-1)

    sort_var = torch.sort(std / mean)[1]
    avg = mean.reshape(1, -1)
    ref = {}
    ref['avg'] = avg
    ref['sort_var'] = sort_var
    torch.save(ref, 'ref_res34.pt')

    y = np.array(cv)
    x = np.arange(cv.shape[0])
    # create two subplots with the shared x and y axes
    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.fill_between(x, 0, y, alpha=1)
    bwith = 2
    ax.set_ylabel('Coefficient of Variation', fontsize=12)
    ax.set_xlabel('Index', fontsize=12)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # ax.label_outer()
    plt.tick_params(labelsize=12)
    plt.grid(color='black', linestyle='-.', linewidth=1)
    fig.autofmt_xdate()
    plt.show()
