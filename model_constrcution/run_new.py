import os

# 必须放在最前面
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

import torch
import numpy as np
from DNN_models.Complex_MTASS_Solver import Complex_MTASS_model


TRAIN_H5 = "/work104/lishuailong/dataset/MTASS-dataset-16K/train_new_ready.h5"
DEV_H5 = "/work104/lishuailong/dataset/MTASS-dataset-16K/dev_new_ready.h5"


TEST_DATA_DIR = "/work104/lishuailong/dataset/MTASS-dataset-16K/test"

MODEL_SAVE_DIR = "./model_parameters"
RESUME_PATH = None 


LR = 0.001
EPOCHS = 50
BATCH_SIZE = 128  
ALPHA = 0.01         
GRAD_CLIP = 20.0     


MODEL_TRAINING = 1 
MODEL_TESTING = 1  

if __name__ == '__main__':
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # --- 训练模式 ---
    if MODEL_TRAINING == 1:
        print(">>> 启动训练流程 (复数域多任务分离)...")
        Complex_MTASS_model.train_model(
            train_h5=TRAIN_H5,
            dev_h5=DEV_H5,
            learning_rate=LR,
            num_epochs=EPOCHS,
            mini_batch_size=BATCH_SIZE,
            alpha=ALPHA,
            resume_path=RESUME_PATH,
            grad_clip=GRAD_CLIP
        )

    # --- 测试模式 ---
    if MODEL_TESTING == 1:
        print(">>> 启动全长原始 Wav 音频评估测试...")
        latest_model = os.path.join(MODEL_SAVE_DIR, f"epoch{EPOCHS-1}.pth")
        
        if os.path.exists(latest_model):
            Complex_MTASS_model.test_model(
                test_dir=TEST_DATA_DIR, 
                model_path=latest_model,
                num_save=10
            )
        else:
            print(f"错误: 未找到模型文件 {latest_model}")