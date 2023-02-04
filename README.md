# Diffusion-model-demo
A demo of diffusion model

# Detail
## Directory Structure
```
Diffusion-model-demo
└── __pycache__
└── Loss.py         # diffusion_loss function
└── README.md       # readme.md
└── inference.py    # inference and visualize
└── model.py        # diffusion model composed of MLP
└── test.py         # some test code ,include the process of diffusion \ 
                    # it is useless for training and inferencing
└── train.py        # train code
```
## Environment
There are a part of packages which this project requirs.
- pytorch-1.11.0
- matplotlib-3.5.1
- numpy-1.21.5
- scikit-learn-1.2.1
- python-3.9.12
## Run
It is recommended to use the IDE for debugging. Here are some suggestions:
- `test.py` shows the process of diffusion, if you don't want to see it, please ignore this file.
- You can run `train.py` to generate models, and modify the save_path in it.
- After `train.py`, `inference.py` shows the reverse diffusion process--how to generate the target distribution by standard normal distribution.
- `model.py` is diffusion model which is composed of MLP, you can try another network.
- `Loss.py` stipulates that the network prediction is noise, you can modify the target.
## Result
- ![diffusion process](https://raw.githubusercontent.com/Zhoujp-Runner/Diffusion-model-demo/main/result/diffusion_process.png?token=GHSAT0AAAAAAB6GXOAMUB4QILU6KD4GNX6CY66M3QQ)
- ![inference process](https://raw.githubusercontent.com/Zhoujp-Runner/Diffusion-model-demo/main/result/inference.png?token=GHSAT0AAAAAAB6GXOAN7V2MJZGZTMO6EUTGY66M4DA)
# Notes
- This project is based on the instructional video, and the url is here.**[[bilibili/deep_thought](https://www.bilibili.com/video/BV1b541197HX/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=e3780c93bbfab1295672c1a3f1be54d5)]**
- The annotation is in Chinese.
