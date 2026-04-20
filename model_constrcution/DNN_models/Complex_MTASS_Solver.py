
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter 
import h5py
from thop import profile
from thop import clever_format
import wave
import struct
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import scipy.signal as signal
import os
import gc
import datetime
import time
import random
from tqdm import tqdm
import librosa
import sys
sys.path.append("..")
from utils.utils_library import *
from DNN_models.Complex_MTASS import *


class MTASSH5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path

        with h5py.File(h5_path, 'r') as f:
            self.length = f['X1'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'h5_file'):
            # 修正参数名：rdcc_nslots
            self.h5_file = h5py.File(
                self.h5_path, 'r', 
                rdcc_nbytes=1024**3,  
                rdcc_nslots=10007,    
                libver='latest', 
                swmr=True
            )
            
        f = self.h5_file
        

        x1 = torch.from_numpy(f['X1'][idx])
        y1 = torch.from_numpy(f['Y1'][idx])
        y2 = torch.from_numpy(f['Y2'][idx])
        y3 = torch.from_numpy(f['Y3'][idx])
        r1 = torch.from_numpy(f['R1'][idx])
        r2 = torch.from_numpy(f['R2'][idx])
        r3 = torch.from_numpy(f['R3'][idx])
        
        return x1, [y1, y2, y3], [r1, r2, r3]

# -----------------------------------------------------------------------------------------------------------------------------
# * Class:
#     Complex_MTASS_model---Implements a Complex-domain MTASS model for speech, noise and music separation 
# 
# * Note:
#   The Complex MTASS model takes the mixture Mag feratures as the inputs 
#   and outputs the complex ratio masks (cRMs).
#   In this model, 8 sub-bands are divided and performed the multi-scale analysis.
#   
#
# * Copyright and Authors:
#    Writen by Mr. Wind at Harbin Institute of Technology, Shenzhen.
#    Contact Email: zhanglu_wind@163.com
# -----------------------------------------------------------------------------------------------------------------------------

class Complex_MTASS_model:

    #-------------------------------------------------------------------------------------------------------------------
    # * Functions:
    #     model_description()-- print the description of model and the information of training data 
    #        * Arguments:
    #            * train_datain_path1 -- train data path of input
    #            * train_datain_list1 -- train data list of input
    #            * dev_datain_path1 -- dev data path of input
    #            * dev_datain_list1 -- dev data list of output
    #            * mini_batch_size -- the size of each mini_batch
    #        * Returns:
    #            * m_x1_train -- input feature size, shape [0]
    #            * n_x1_train -- input feature size, shape [1]
    #            * num_minibatches_train -- the toal numbers of training data
    #            * num_minibatches_dev -- the total numbers of dev data
    #
    #---------------------------------------------------------------------------------------------------------------------

    def model_description(train_datain_path1,train_datain_list1,dev_datain_path1,dev_datain_list1,mini_batch_size):
        ### START CODE HERE ###
        print('The Complex MTASS learning structure (Mag_to_Com, Residual Compensation, F-MSE+T-SNR) is : 257+ComplexMSTCN(15)+3*GTCN(5,8)+(514,514,514)')
        print('The Complex MTASS model is trained to separate three targets!') 
        print('The sizes of each train/dev file are as follows:')
        num_minibatches_train = 0
        num_minibatches_dev = 0        
        for train_datain in train_datain_list1:
            path = train_datain_path1 + os.sep + train_datain
            data = np.load(path)
            # print('n_x_train', n_x_train)
            num_minibatches_train += math.floor(data.shape[0] / mini_batch_size)
            split_sentence_len = data.shape[2] 

        for dev_datain in dev_datain_list1:
            path = dev_datain_path1 + os.sep + dev_datain
            data = np.load(path)
            # print('n_x_dev', n_x_dev)
            num_minibatches_dev += math.floor(data.shape[0] / mini_batch_size)
                
        print('The mini_batch size is:', mini_batch_size)
        print('num_minibatches_train:', num_minibatches_train)
        print('num_minibatches_dev:', num_minibatches_dev)
    
        del data
        gc.collect()
        return num_minibatches_train, num_minibatches_dev, split_sentence_len 


    # -------------------------------------------------------------------------------------------------------
    #
    # * Function:     
    #    load_dataset(args,path)-- generate the binary data (.npy) for training and validation, 
    #                              and the generated data will be saved in file 'train_data' or 'dev_data'.
    #    * Arguments:
    #        * args -- Configure data loading for 'train' or 'dev'
    #        * path -- The file path of original .wav dataset
    #        * file_num -- The total numbers of choped files
    #    * Returns:
    #        * None
    # -------------------------------------------------------------------------------------------------------

    def load_dataset(args,path,file_num):
        
        dataset_name = args
        MIX_DATASET_PATH = path + os.sep + 'mixture'
        SPEECH_DATASET_PATH = path + os.sep + 'speech'
        NOISE_DATASET_PATH = path + os.sep + 'noise'
        MUSIC_DATASET_PATH = path + os.sep + 'music'
        print(path)

        mixture_folders_list = os.listdir(MIX_DATASET_PATH)  # get the dirname of each kind of audio
        file_list = np.array(mixture_folders_list)
        # np.save('./train_data/noisy_folders_list.npy', file_list)
        chop_file_num = file_num

        frame_size = 512
        frame_shift = int(frame_size / 2)
        input_feature_size = 257

        winfunc = signal.windows.hamming(frame_size)
        mixture_datapath_list = []
        speech_datapath_list = []
        noise_datapath_list = []
        music_datapath_list = []
        
        print("Loading %s data path in a list..." %(args))
        for mixture_folder in tqdm(mixture_folders_list):
            file_code = mixture_folder[7:]
            # print("file_code is:", file_code)
            
            mixture_folder_path = MIX_DATASET_PATH + os.sep + mixture_folder
            mixture_file_list = os.listdir(mixture_folder_path)
            mixture_wav_file_path = mixture_folder_path + os.sep + mixture_file_list[0]
            # print(mixture_wav_file_path)
            mixture_datapath_list.append(mixture_wav_file_path)
            
            speech_folder_path = SPEECH_DATASET_PATH + os.sep + 'speech' + file_code
            speech_file_list = os.listdir(speech_folder_path)
            speech_wav_file_path = speech_folder_path + os.sep + speech_file_list[0]
            # print(speech_wav_file_path)
            speech_datapath_list.append(speech_wav_file_path)

            noise_folder_path = NOISE_DATASET_PATH + os.sep + 'noise' + file_code
            noise_file_list = os.listdir(noise_folder_path)
            noise_wav_file_path = noise_folder_path + os.sep + noise_file_list[0]
            # print(noise_wav_file_path)
            noise_datapath_list.append(noise_wav_file_path)

            music_folder_path = MUSIC_DATASET_PATH + os.sep + 'music' + file_code
            music_file_list = os.listdir(music_folder_path)
            music_wav_file_path = music_folder_path + os.sep + music_file_list[0]
            # print(music_wav_file_path)
            music_datapath_list.append(music_wav_file_path)
            

        print('\n')
        print("Extracting features...")
        permutation = np.random.permutation(len(mixture_datapath_list))
        
        file_count = 0
        chop_count = 0
        chop_num = math.ceil(len(mixture_datapath_list)/chop_file_num)
    
        dataset_input_X1 = []
        dataset_input_X2 = []
        dataset_input_X3 = []
        dataset_output_Y1 = []
        dataset_output_Y2 = []
        ii_end = permutation[-1]
        
        for ii in tqdm(permutation):

            # Extracting mixture input features
            mixture_sound, mixture_samplerate = wav_read(mixture_datapath_list[ii])
            mixture_split = enframe(mixture_sound, frame_size, frame_shift, winfunc)
            mixture_frequence = compute_fft(mixture_split, frame_size)
            mixture_frequence_split = RI_split(mixture_frequence, mixture_frequence.shape[0])
            mixture_input_feature_1 = mixture_frequence_split
            mixture_input_feature_2 = mixture_split
            
            # Extracting clean speech features
            clean_speech, speech_samplerate = wav_read(speech_datapath_list[ii])
            speech_split = enframe(clean_speech, frame_size, frame_shift, winfunc)
            speech_frequence = compute_fft(speech_split, frame_size)
            speech_frequence_split = RI_split(speech_frequence, speech_frequence.shape[0])
            speech_output_feature_1 = speech_frequence_split
            speech_output_feature_2 = speech_split
            
            # Extracting ideal noise features
            ideal_noise, noise_samplerate = wav_read(noise_datapath_list[ii])
            noise_split = enframe(ideal_noise, frame_size, frame_shift, winfunc)
            noise_frequence = compute_fft(noise_split, frame_size)
            noise_frequence_split = RI_split(noise_frequence, noise_frequence.shape[0])
            noise_output_feature_1 = noise_frequence_split
            noise_output_feature_2 = noise_split
            
            # Extracting clean music features
            ideal_music, music_samplerate = wav_read(music_datapath_list[ii])
            music_split = enframe(ideal_music, frame_size, frame_shift, winfunc)
            music_frequence = compute_fft(music_split, frame_size)
            music_frequence_split = RI_split(music_frequence, music_frequence.shape[0])
            music_output_feature_1 = music_frequence_split
            music_output_feature_2 = music_split
            
    
            # Appending feature array
            if file_count == 0:
                dataset_input_X1 = mixture_input_feature_1
                dataset_input_X2 = mixture_input_feature_2
                dataset_output_Y1 = speech_output_feature_1
                dataset_output_Y2 = noise_output_feature_1
                dataset_output_Y3 = music_output_feature_1
                dataset_output_S1 = speech_output_feature_2
                dataset_output_S2 = noise_output_feature_2
                dataset_output_S3 = music_output_feature_2
            else:
                dataset_input_X1 = np.hstack((dataset_input_X1, mixture_input_feature_1)) # dataset_input_X1.shape = [514,?]
                dataset_input_X2 = np.hstack((dataset_input_X2, mixture_input_feature_2)) # dataset_input_X2.shape = [512,?]
                dataset_output_Y1 = np.hstack((dataset_output_Y1, speech_output_feature_1)) # dataset_output_Y1.shape = [514,?]
                dataset_output_Y2 = np.hstack((dataset_output_Y2, noise_output_feature_1)) # dataset_output_Y2.shape = [514,?]
                dataset_output_Y3 = np.hstack((dataset_output_Y3, music_output_feature_1)) # dataset_output_Y3.shape = [514,?]
                dataset_output_S1 = np.hstack((dataset_output_S1, speech_output_feature_2)) # dataset_output_S1.shape = [512,?]
                dataset_output_S2 = np.hstack((dataset_output_S2, noise_output_feature_2)) # dataset_output_S2.shape = [512,?]
                dataset_output_S3 = np.hstack((dataset_output_S3, music_output_feature_2)) # dataset_output_S3.shape = [512,?]
    
            
            # Saving feature array
            if dataset_name == 'train':
                if (file_count == (chop_num - 1)) or (ii == ii_end):
                    if not os.path.exists(os.path.dirname('./train_data/data_in_X1/')):
                        os.makedirs(os.path.dirname('./train_data/data_in_X1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_in_X2/')):
                        os.makedirs(os.path.dirname('./train_data/data_in_X2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y1/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y2/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_Y3/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_Y3/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S1/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S1/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S2/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S2/'))
                    if not os.path.exists(os.path.dirname('./train_data/data_out_S3/')):
                        os.makedirs(os.path.dirname('./train_data/data_out_S3/'))
                    
                    filename_in_X1 = './train_data/data_in_X1' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_in_X2 = './train_data/data_in_X2' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_out_Y1 = './train_data/data_out_Y1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_Y2 = './train_data/data_out_Y2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_Y3 = './train_data/data_out_Y3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    filename_out_S1 = './train_data/data_out_S1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_S2 = './train_data/data_out_S2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_S3 = './train_data/data_out_S3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    
                    np.save(filename_in_X1, dataset_input_X1)
                    np.save(filename_in_X2, dataset_input_X2)
                    np.save(filename_out_Y1, dataset_output_Y1)
                    np.save(filename_out_Y2, dataset_output_Y2)
                    np.save(filename_out_Y3, dataset_output_Y3)
                    np.save(filename_out_S1, dataset_output_S1)
                    np.save(filename_out_S2, dataset_output_S2)
                    np.save(filename_out_S3, dataset_output_S3)
                    
                    chop_count = chop_count + 1
                    file_count = 0
                else:
                    file_count = file_count + 1
            
            if dataset_name == 'dev':
                if (file_count == (chop_num - 1)) or (ii == ii_end):
                    if not os.path.exists(os.path.dirname('./dev_data/data_in_X1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_in_X1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_in_X2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_in_X2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_Y3/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_Y3/'))                    
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S1/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S1/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S2/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S2/'))
                    if not os.path.exists(os.path.dirname('./dev_data/data_out_S3/')):
                        os.makedirs(os.path.dirname('./dev_data/data_out_S3/')) 
                        
                    filename_in_X1 = './dev_data/data_in_X1' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_in_X2 = './dev_data/data_in_X2' + os.sep + "data" + str(chop_count) + "_mix" + ".npy"
                    filename_out_Y1 = './dev_data/data_out_Y1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_Y2 = './dev_data/data_out_Y2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_Y3 = './dev_data/data_out_Y3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    filename_out_S1 = './dev_data/data_out_S1' + os.sep + "data" + str(chop_count) + "_speech" + ".npy"
                    filename_out_S2 = './dev_data/data_out_S2' + os.sep + "data" + str(chop_count) + "_noise" + ".npy"
                    filename_out_S3 = './dev_data/data_out_S3' + os.sep + "data" + str(chop_count) + "_music" + ".npy"
                    
                    np.save(filename_in_X1, dataset_input_X1)
                    np.save(filename_in_X2, dataset_input_X2)
                    np.save(filename_out_Y1, dataset_output_Y1)
                    np.save(filename_out_Y2, dataset_output_Y2)
                    np.save(filename_out_Y3, dataset_output_Y3)
                    np.save(filename_out_S1, dataset_output_S1)
                    np.save(filename_out_S2, dataset_output_S2)
                    np.save(filename_out_S3, dataset_output_S3)
                    
                    chop_count = chop_count + 1
                    file_count = 0
                else:
                    file_count = file_count + 1


   
    # -------------------------------------------------------------------------------------------------------
    # *Function: 
    #     split_and_reshape_data()- 1. Covert the enframed and overlapped time-domain data to orignal waveform data
    #                               2. Split the data to fixed sentence length
    #                               3. Reshape the features
    # *Arguments:
    #    * args -- Configure data processing for 'train' or 'dev'
    #    * rejoin_sentence_len -- The split sentence length
    # * Return:
    #    * None
    # ------------------------------------------------------------------------------------------------------        

    def split_and_reshape_data(args,rejoin_sentence_len):
        
        if args == 'train':
            train_datain_path1 = './train_data/data_in_X1' # Mixture RI
            train_dataout_path1 = './train_data/data_out_S1' # Speech time (enframed)
            train_dataout_path2 = './train_data/data_out_S2' # Noise time (enframed)
            train_dataout_path3 = './train_data/data_out_S3' # Music time (enframed)
            train_dataout_path4 = './train_data/data_out_Y1' # Speech RI
            train_dataout_path5 = './train_data/data_out_Y2' # Noise RI
            train_dataout_path6 = './train_data/data_out_Y3' # Music RI

            resaved_train_datain_path = './train_data/tmp/data_in_X1' # Mixture RI
            resaved_train_dataout_path1 = './train_data/tmp/data_out_S1' # Speech time (enframed)
            resaved_train_dataout_path2 = './train_data/tmp/data_out_S2' # Noise time (enframed)
            resaved_train_dataout_path3 = './train_data/tmp/data_out_S3' # Music time (enframed)
            resaved_train_dataout_path4 = './train_data/tmp/data_out_Y1' # Speech RI
            resaved_train_dataout_path5 = './train_data/tmp/data_out_Y2' # Noise RI
            resaved_train_dataout_path6 = './train_data/tmp/data_out_Y3' # Music RI
            resaved_train_dataout_path7 = './train_data/tmp/data_out_R1' # Speech time (?, sen_len)
            resaved_train_dataout_path8 = './train_data/tmp/data_out_R2' # Noise time (?, sen_len)
            resaved_train_dataout_path9 = './train_data/tmp/data_out_R3' # Music time (?, sen_len)  
            if not os.path.exists(resaved_train_datain_path):
                os.makedirs(resaved_train_datain_path)
            if not os.path.exists(resaved_train_dataout_path1):
                os.makedirs(resaved_train_dataout_path1)
            if not os.path.exists(resaved_train_dataout_path2):
                os.makedirs(resaved_train_dataout_path2)
            if not os.path.exists(resaved_train_dataout_path3):
                os.makedirs(resaved_train_dataout_path3)
            if not os.path.exists(resaved_train_dataout_path4):
                os.makedirs(resaved_train_dataout_path4)
            if not os.path.exists(resaved_train_dataout_path5):
                os.makedirs(resaved_train_dataout_path5)
            if not os.path.exists(resaved_train_dataout_path6):
                os.makedirs(resaved_train_dataout_path6)
            if not os.path.exists(resaved_train_dataout_path7):
                os.makedirs(resaved_train_dataout_path7)
            if not os.path.exists(resaved_train_dataout_path8):
                os.makedirs(resaved_train_dataout_path8)
            if not os.path.exists(resaved_train_dataout_path9):
                os.makedirs(resaved_train_dataout_path9)
                            
            train_datain_list1 = os.listdir(train_datain_path1)
            
            print('Split and Reshape Train Data...')
            for train_datain in tqdm(train_datain_list1):
            
                file_code = train_datain[0:-8]
                frame_shift = 256
                # print(file_code)
                in_path = train_datain_path1 + os.sep + train_datain
                X1 = np.load(in_path)
    
                path1 = train_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                Y1 = np.load(path1)
                path2 = train_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                Y2 = np.load(path2)
                path3 = train_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                Y3 = np.load(path3)
                path4 = train_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                Y4 = np.load(path4)
                path5 = train_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                Y5 = np.load(path5)
                path6 = train_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                Y6 = np.load(path6)
                
                X1 = X1.T # X1.shape = [?,514](mixture)    
                Y1 = Y1.T # Y1.shape = [?,512](speech time)
                Y2 = Y2.T # Y2.shape = [?,512](noise time)
                Y3 = Y3.T # Y3.shape = [?,512](music time)
                Y4 = Y4.T # Y4.shape = [?,514](speech RI)
                Y5 = Y5.T # Y5.shape = [?,514](noise RI)
                Y6 = Y6.T # Y6.shape = [?,514](music RI)
    
                del_row = Y1.shape[0]%rejoin_sentence_len
                if del_row == 0:
                    X1 = np.reshape(X1[:,:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_sen,514,re_sen_len]
                    Y1 = np.reshape(Y1[:,:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_sen,512,re_sen_len]
                    Y2 = np.reshape(Y2[:,:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_sen,512,re_sen_len]
                    Y3 = np.reshape(Y3[:,:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_sen,512,re_sen_len]
                    Y4 = np.reshape(Y4[:,:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_sen,514,re_sen_len]
                    Y5 = np.reshape(Y5[:,:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_sen,514,re_sen_len]
                    Y6 = np.reshape(Y6[:,:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_sen,514,re_sen_len]
                else:
                    X1 = np.reshape(X1[:(-del_row),:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_sen,514,re_sen_len]
                    Y1 = np.reshape(Y1[:(-del_row),:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_sen,512,re_sen_len]
                    Y2 = np.reshape(Y2[:(-del_row),:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_sen,512,re_sen_len]
                    Y3 = np.reshape(Y3[:(-del_row),:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_sen,512,re_sen_len]
                    Y4 = np.reshape(Y4[:(-del_row),:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_sen,514,re_sen_len]
                    Y5 = np.reshape(Y5[:(-del_row),:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_sen,514,re_sen_len]
                    Y6 = np.reshape(Y6[:(-del_row),:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_sen,514,re_sen_len]    
                Y7 = overlap_add_batch(Y1,frame_shift) # Y7.shape = [num_sen,256*(re_sen_len-1)]
                Y8 = overlap_add_batch(Y2,frame_shift) # Y8.shape = [num_sen,256*(re_sen_len-1)]
                Y9 = overlap_add_batch(Y3,frame_shift) # Y9.shape = [num_sen,256*(re_sen_len-1)]
    
                resaved_train_filename1 = resaved_train_datain_path + os.sep + file_code + "_mixture" + ".npy"
                np.save(resaved_train_filename1, X1) # X1, mixture RI
                
                resaved_train_filename2 = resaved_train_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename2, Y1) # Y1, speech frame                                
                resaved_train_filename3 = resaved_train_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename3, Y2) # Y2, noise frame
                resaved_train_filename4 = resaved_train_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename4, Y3) # Y3, music frame
                
                resaved_train_filename5 = resaved_train_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename5, Y4) # Y4, speech RI                                
                resaved_train_filename6 = resaved_train_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename6, Y5) # Y5, noise RI
                resaved_train_filename7 = resaved_train_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename7, Y6) # Y6, music RI
                
                resaved_train_filename8 = resaved_train_dataout_path7 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_train_filename8, Y7) # Y7, speech sentence                                
                resaved_train_filename9 = resaved_train_dataout_path8 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_train_filename9, Y8) # Y8, noise sentence
                resaved_train_filename10 = resaved_train_dataout_path9 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_train_filename10, Y9) # Y9, music sentence

        
        if args == 'dev':
            dev_datain_path1 = './dev_data/data_in_X1' # Mixture RI
            dev_dataout_path1 = './dev_data/data_out_S1' # Speech time (enframed)
            dev_dataout_path2 = './dev_data/data_out_S2' # Noise time (enframed)
            dev_dataout_path3 = './dev_data/data_out_S3' # Music time (enframed)
            dev_dataout_path4 = './dev_data/data_out_Y1' # Speech RI
            dev_dataout_path5 = './dev_data/data_out_Y2' # Noise RI
            dev_dataout_path6 = './dev_data/data_out_Y3' # Music RI
    
            resaved_dev_datain_path = './dev_data/tmp/data_in_X1' # Mixture RI
            resaved_dev_dataout_path1 = './dev_data/tmp/data_out_S1' # Speech time (enframed)
            resaved_dev_dataout_path2 = './dev_data/tmp/data_out_S2' # Noise time (enframed)
            resaved_dev_dataout_path3 = './dev_data/tmp/data_out_S3' # Music time (enframed)
            resaved_dev_dataout_path4 = './dev_data/tmp/data_out_Y1' # Speech RI
            resaved_dev_dataout_path5 = './dev_data/tmp/data_out_Y2' # Noise RI
            resaved_dev_dataout_path6 = './dev_data/tmp/data_out_Y3' # Music RI
            resaved_dev_dataout_path7 = './dev_data/tmp/data_out_R1' # Speech time (?, sen_len)
            resaved_dev_dataout_path8 = './dev_data/tmp/data_out_R2' # Noise time (?, sen_len)
            resaved_dev_dataout_path9 = './dev_data/tmp/data_out_R3' # Music time (?, sen_len)  
            if not os.path.exists(resaved_dev_datain_path):
                os.makedirs(resaved_dev_datain_path)
            if not os.path.exists(resaved_dev_dataout_path1):
                os.makedirs(resaved_dev_dataout_path1)
            if not os.path.exists(resaved_dev_dataout_path2):
                os.makedirs(resaved_dev_dataout_path2)
            if not os.path.exists(resaved_dev_dataout_path3):
                os.makedirs(resaved_dev_dataout_path3)
            if not os.path.exists(resaved_dev_dataout_path4):
                os.makedirs(resaved_dev_dataout_path4)
            if not os.path.exists(resaved_dev_dataout_path5):
                os.makedirs(resaved_dev_dataout_path5)
            if not os.path.exists(resaved_dev_dataout_path6):
                os.makedirs(resaved_dev_dataout_path6)
            if not os.path.exists(resaved_dev_dataout_path7):
                os.makedirs(resaved_dev_dataout_path7)
            if not os.path.exists(resaved_dev_dataout_path8):
                os.makedirs(resaved_dev_dataout_path8)
            if not os.path.exists(resaved_dev_dataout_path9):
                os.makedirs(resaved_dev_dataout_path9)  
            
            dev_datain_list1 = os.listdir(dev_datain_path1)

            print('Split and Reshape Dev Data...')
            for dev_datain in tqdm(dev_datain_list1):
            
                file_code = dev_datain[0:-8]
                frame_shift = 256
                # print(file_code)
                in_path = dev_datain_path1 + os.sep + dev_datain
                X1 = np.load(in_path)
    
                path1 = dev_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                Y1 = np.load(path1)
                path2 = dev_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                Y2 = np.load(path2)
                path3 = dev_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                Y3 = np.load(path3)
                path4 = dev_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                Y4 = np.load(path4)
                path5 = dev_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                Y5 = np.load(path5)
                path6 = dev_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                Y6 = np.load(path6)
                
                X1 = X1.T # X1.shape = [?,514](mixture)    
                Y1 = Y1.T # Y1.shape = [?,512](speech)
                Y2 = Y2.T # Y2.shape = [?,512](noise)
                Y3 = Y3.T # Y3.shape = [?,512](music)
                Y4 = Y4.T # Y4.shape = [?,514](speech)
                Y5 = Y5.T # Y5.shape = [?,514](noise)
                Y6 = Y6.T # Y6.shape = [?,514](music)
    
                del_row = Y1.shape[0]%rejoin_sentence_len
                if del_row == 0:
                    X1 = np.reshape(X1[:,:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_spl,514,re_sen_len]
                    Y1 = np.reshape(Y1[:,:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_spl,512,re_sen_len]
                    Y2 = np.reshape(Y2[:,:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_spl,512,re_sen_len]
                    Y3 = np.reshape(Y3[:,:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_spl,512,re_sen_len]
                    Y4 = np.reshape(Y4[:,:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_spl,514,re_sen_len]
                    Y5 = np.reshape(Y5[:,:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_spl,514,re_sen_len]
                    Y6 = np.reshape(Y6[:,:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_spl,514,re_sen_len]
                else:
                    X1 = np.reshape(X1[:(-del_row),:],(-1,rejoin_sentence_len,X1.shape[1])).transpose((0,2,1)) # X1.shape=[num_spl,514,re_sen_len]
                    Y1 = np.reshape(Y1[:(-del_row),:],(-1,rejoin_sentence_len,Y1.shape[1])).transpose((0,2,1)) # Y1.shape=[num_spl,512,re_sen_len]
                    Y2 = np.reshape(Y2[:(-del_row),:],(-1,rejoin_sentence_len,Y2.shape[1])).transpose((0,2,1)) # Y2.shape=[num_spl,512,re_sen_len]
                    Y3 = np.reshape(Y3[:(-del_row),:],(-1,rejoin_sentence_len,Y3.shape[1])).transpose((0,2,1)) # Y3.shape=[num_spl,512,re_sen_len]
                    Y4 = np.reshape(Y4[:(-del_row),:],(-1,rejoin_sentence_len,Y4.shape[1])).transpose((0,2,1)) # Y4.shape=[num_spl,514,re_sen_len]
                    Y5 = np.reshape(Y5[:(-del_row),:],(-1,rejoin_sentence_len,Y5.shape[1])).transpose((0,2,1)) # Y5.shape=[num_spl,514,re_sen_len]
                    Y6 = np.reshape(Y6[:(-del_row),:],(-1,rejoin_sentence_len,Y6.shape[1])).transpose((0,2,1)) # Y6.shape=[num_spl,514,re_sen_len]    
                Y7 = overlap_add_batch(Y1,frame_shift) # Y7.shape = [num_spl,256*(re_sen_len-1)]
                Y8 = overlap_add_batch(Y2,frame_shift) # Y8.shape = [num_spl,256*(re_sen_len-1)]
                Y9 = overlap_add_batch(Y3,frame_shift) # Y9.shape = [num_spl,256*(re_sen_len-1)]
    
                resaved_dev_filename1 = resaved_dev_datain_path + os.sep + file_code + "_mixture" + ".npy"
                np.save(resaved_dev_filename1, X1) # X1, mixture RI
                
                resaved_dev_filename2 = resaved_dev_dataout_path1 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename2, Y1) # Y1, speech frame                                
                resaved_dev_filename3 = resaved_dev_dataout_path2 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename3, Y2) # Y2, noise frame
                resaved_dev_filename4 = resaved_dev_dataout_path3 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename4, Y3) # Y3, music frame
                
                resaved_dev_filename5 = resaved_dev_dataout_path4 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename5, Y4) # Y4, speech RI                                
                resaved_dev_filename6 = resaved_dev_dataout_path5 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename6, Y5) # Y5, noise RI
                resaved_dev_filename7 = resaved_dev_dataout_path6 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename7, Y6) # Y6, music RI
                
                resaved_dev_filename8 = resaved_dev_dataout_path7 + os.sep + file_code + "_speech" + ".npy"
                np.save(resaved_dev_filename8, Y7) # Y7, speech sentence                                
                resaved_dev_filename9 = resaved_dev_dataout_path8 + os.sep + file_code + "_noise" + ".npy"
                np.save(resaved_dev_filename9, Y8) # Y8, noise sentence
                resaved_dev_filename10 = resaved_dev_dataout_path9 + os.sep + file_code + "_music" + ".npy"
                np.save(resaved_dev_filename10, Y9) # Y9, music sentence
        

    
    def compute_out_cost(Z1, Z2, Z3, Y1, Y2, Y3, Y4, Y5, Y6, alpha):
        """
        Z1-Z3: 模型输出 (Speech, Music, Noise)
        Y1-Y3: 频域标签 RI (对应 Speech, Music, Noise)
        Y4-Y6: 时域标签 Wave (对应 Speech, Music, Noise)
        """
        win_len, win_inc, fft_len = 512, 256, 512
        mse_cost = torch.nn.MSELoss()
        
        # --- 1. MSE Total (频域) ---
        mse_total = mse_cost(Z1, Y1) + mse_cost(Z2, Y2) + mse_cost(Z3, Y3)
        
        # --- 2. SNR Total (时域) ---

        target_len = Y4.size(1)
        

        Z1_t = Complex_MTASS_model.Inverse_STFT(Z1, win_len, win_inc, fft_len, expected_length=target_len)
        Z2_t = Complex_MTASS_model.Inverse_STFT(Z2, win_len, win_inc, fft_len, expected_length=target_len)
        Z3_t = Complex_MTASS_model.Inverse_STFT(Z3, win_len, win_inc, fft_len, expected_length=target_len)
        

        snr_total = Complex_MTASS_model.SNR_cost(Z1_t, Y4) + \
                    Complex_MTASS_model.SNR_cost(Z2_t, Y5) + \
                    Complex_MTASS_model.SNR_cost(Z3_t, Y6)
        

        total_loss = mse_total + alpha * snr_total
        
        return {
            'total': total_loss,
            'mse': mse_total,
            'snr': snr_total
        }

    def SNR_cost(Z1,Y1,eps=1e-8):
        # Z1.shape=[-1,sen_len]
        # Y1.shape=[-1,sen_len]

        snr = torch.sum(Y1**2, dim=1, keepdim=True) / (torch.sum((Z1 - Y1)**2, dim=1, keepdim=True)+eps)
        loss = -10*torch.log10(snr + eps).mean()
        
        return loss


    def Inverse_STFT(inputs, win_len, win_hop, fft_len, expected_length=None):
        # inputs.shape = [B, 514, T_frames] (前257维实部, 后257维虚部)
        cutoff = fft_len // 2 + 1
        real_part = inputs[:, :cutoff, :]
        imag_part = inputs[:, cutoff:, :]

        complex_spec = torch.complex(real_part, imag_part)
        

        istft_window = torch.hamming_window(win_len, periodic=True).to(inputs.device)
        
        reconstruction = torch.istft(
            complex_spec,
            n_fft=fft_len,
            hop_length=win_hop,
            win_length=win_len,
            window=istft_window,
            center=True,          
            normalized=False,
            onesided=True,
            length=expected_length, 
            return_complex=False 
        )
        return reconstruction
    # ---------------------------------------------------------------------------------------
    # * Function:
    #     reshape_test_data(datain)-Reshape the extracted features (test) 
    #                               to the shapes that testing Complex-MTASSNet model needs
    #
    #   * Arguments:
    #    * datain1 -- input test data for reshaping
    #   * Returns:
    #    * data_input1 -- reshaped output data
    #
    # -----------------------------------------------------------------------------------------
    
    def reshape_test_data(datain1):
        # datain1.shape = [514,-1]
        data_input1 = np.reshape(datain1,(1,datain1.shape[0],datain1.shape[1])) # data_input1.shape = (1,514,-1)
        
        return data_input1
    
    # ---------------------------------------------------------------------------------------
    # * Function:
    #     post_processing()-Post-processing for model inference
    #
    #   * Arguments:
    #    * separated_frequency -- the separated RI for each track
    #   * Returns:
    #    * separated_time -- output of the post-processing module 
    #
    # -----------------------------------------------------------------------------------------
        
    def post_processing(separated_frequency,frame_size):
        # separated_frequency.shape = (1,514,-1)
        separated_frequency = separated_frequency.transpose((2,1,0)) # enhanced_Mag.shape = (-1,514,1)
        sample_num = separated_frequency.shape[0]
        separated_frequency = np.reshape(separated_frequency, (sample_num, -1))  # separated_frequency.shape = (sample_num,514)
        separated_frequency = separated_frequency.T  # separated_frequency.shape = (514,sample_num)
        separated_frequency = RI_interpolation(separated_frequency,separated_frequency.shape[0])
        separated_time = compute_ifft(separated_frequency, frame_size)   
                 
        return separated_time    
    
    # -------------------------------------------------------------------------------------------
    # * Function:
    #     train_model()---Train MSTCN_SE-MT model for monaural speech enhancement
    #        * Regularization methods: Batch normalization, Dropout 
    #        * Optimization method: AdamOptimize
    #
    #    * Arguments:
    #        * learning_rate -- learning rate of the optimization
    #        * num_epochs -- number of epochs of the optimization loop
    #        * minibatch_size -- size of a minibatch
    #        * print_cost -- True to print cost
    #        * validation -- True to compute mse cost on dev dataset
    #        * show_model_size -- calculate the trainable parameters of model
    #        * gradient_clip -- True to perform gradient clipping
    #        * continue_train -- True to continue train the model by loading the saved model
    #        * set_num_workers -- number of workers for dataloader
    #        * pin_memory -- True to pin memory for dataloader
    #
    #    * Returns:
    #        * None
    # ------------------------------------------------------------------------------------------
    
    def train_model(train_h5, dev_h5, learning_rate=0.001, num_epochs=50, mini_batch_size=96, 
                    alpha=0.01, resume_path=None, grad_clip=20.0):
        
        # --- 核心多卡设置 ---
        # 1. 初始化模型并移动到 GPU
        net = Complex_MTASS()
        
        if torch.cuda.device_count() > 1:
            print(f">>> 检测到 {torch.cuda.device_count()} 张显卡，启动多卡并行训练...")
            net = nn.DataParallel(net)
        
        net = net.cuda()

        # 3. 如果是 DataParallel，加载权重时需要注意 key 值的匹配
        if resume_path and os.path.exists(resume_path):
            print(f"--- 正在加载断点权重: {resume_path} ---")
            # 兼容性加载：如果保存的是 net.module.state_dict()
            state_dict = torch.load(resume_path)
            if torch.cuda.device_count() > 1:
                net.module.load_state_dict(state_dict)
            else:
                net.load_state_dict(state_dict)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(
            MTASSH5Dataset(train_h5), 
            batch_size=mini_batch_size, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True,
            prefetch_factor=4
        )

        dev_loader = DataLoader(
            MTASSH5Dataset(dev_h5), 
            batch_size=mini_batch_size, 
            shuffle=False, 
            num_workers=12,
            pin_memory=True
        )
        t_writer = SummaryWriter(log_dir='./logs/train')
        v_writer = SummaryWriter(log_dir='./logs/val')
        global_step = 0
        for epoch in range(num_epochs):
            net.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
            for x, y_list, r_list in pbar:
                x = x.cuda()
                y_s, y_m, y_n = [y.cuda() for y in y_list]
                r_s, r_m, r_n = [r.cuda() for r in r_list]

                optimizer.zero_grad()
                z_s, z_m, z_n = net(x) 
                
                loss_d = Complex_MTASS_model.compute_out_cost(z_s, z_m, z_n, y_s, y_m, y_n, r_s, r_m, r_n, alpha)
                
                loss_d['total'].backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip)
                optimizer.step()
                
                t_writer.add_scalar('Loss/Total', loss_d['total'].item(), global_step)
                t_writer.add_scalar('Loss/MSE', loss_d['mse'].item(), global_step)
                t_writer.add_scalar('Loss/SNR', loss_d['snr'].item(), global_step)
                global_step += 1
                pbar.set_postfix(total=f"{loss_d['total'].item():.4f}")

            net.eval()
            v_t, v_m, v_s = 0, 0, 0
            with torch.no_grad():
                for vx, vy_list, vr_list in tqdm(dev_loader, desc=f"Epoch {epoch} [Val]"):
                    vx = vx.cuda()
                    vy_s, vy_m, vy_n = [y.cuda() for y in vy_list]
                    vr_s, vr_m, vr_n = [r.cuda() for r in vr_list]
                    vz_s, vz_m, vz_n = net(vx)
                    v_d = Complex_MTASS_model.compute_out_cost(vz_s, vz_m, vz_n, vy_s, vy_m, vy_n, vr_s, vr_m, vr_n, alpha)
                    v_t += v_d['total'].item(); v_m += v_d['mse'].item(); v_s += v_d['snr'].item()
            
            num_v = len(dev_loader)
            val_writer_step = epoch # 验证集按 epoch 记
            v_writer.add_scalar('Loss/Total', v_t/num_v, val_writer_step)
            v_writer.add_scalar('Loss/MSE', v_m/num_v, val_writer_step)
            v_writer.add_scalar('Loss/SNR', v_s/num_v, val_writer_step)
            
            os.makedirs("./model_parameters", exist_ok=True)
            # torch.save(net.module.state_dict(), f"./model_parameters/epoch{epoch}.pth")
            save_path = f"./model_parameters/epoch{epoch}.pth"
            if isinstance(net, nn.DataParallel):
                torch.save(net.module.state_dict(), save_path)
            else:
                torch.save(net.state_dict(), save_path)
        t_writer.close(); v_writer.close()

    # --------------------------------------------------------------------------------------------------------------------------
    # * Function:
    #     test_model()---Implement inference of the well-trained Complex-MTASSNet model for multi-task audio source separation
    #
    #    * Arguments:
    #        * path -- dataset path
    #
    #    * Returns:
    #        * None
    #
    # --------------------------------------------------------------------------------------------------------------------------
    


    def compute_sdr(ref, est, eps=1e-8):
        """计算信号失真比 (SDR)"""
        # 参考论文公式 (5): 10 * log10(||ref||^2 / ||ref - est||^2) [cite: 175]
        ref_p = torch.sum(ref**2, dim=-1) + eps
        err_p = torch.sum((ref - est)**2, dim=-1) + eps
        return 10 * torch.log10(ref_p / err_p)

 
    def compute_sisdr(ref, est, eps=1e-8):
        """计算尺度不变信号失真比 (SI-SDR)"""
        # 去均值
        ref = ref - torch.mean(ref, dim=-1, keepdim=True)
        est = est - torch.mean(est, dim=-1, keepdim=True)
        # 计算尺度因子 alpha = <est, ref> / ||ref||^2
        alpha = torch.sum(est * ref, dim=-1, keepdim=True) / (torch.sum(ref**2, dim=-1, keepdim=True) + eps)
        target = alpha * ref
        noise = est - target
        return 10 * torch.log10(torch.sum(target**2, dim=-1) / (torch.sum(noise**2, dim=-1) + eps) + eps)

    def test_model(test_dir, model_path, num_save=10, fixed_duration=10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = Complex_MTASS().to(device)

        net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        net.eval()

        TEST_MIX_PATH = os.path.join(test_dir, 'mixture')
        TEST_SPEECH_PATH = os.path.join(test_dir, 'speech')
        TEST_NOISE_PATH = os.path.join(test_dir, 'noise')
        TEST_MUSIC_PATH = os.path.join(test_dir, 'music')

        mixture_folders_list = os.listdir(TEST_MIX_PATH)
        
        metrics = {
            'speech': {'sdr': [], 'sdri': [], 'sisdr': [], 'sisdri': []},
            'music':  {'sdr': [], 'sdri': [], 'sisdr': [], 'sisdri': []},
            'noise':  {'sdr': [], 'sdri': [], 'sisdr': [], 'sisdri': []}
        }

        target_samples = 16000 * fixed_duration
        save_dir_base = "./test_results"
        os.makedirs(save_dir_base, exist_ok=True)
        save_count = 0

        print(f">>> 开始读取原始 Wav 并统一为 {fixed_duration}s 进行评估...")

        with torch.no_grad():
            for mixture_folder in tqdm(mixture_folders_list, desc="Testing"):
                file_code = mixture_folder[7:] 
                
                def get_wav_path(base_dir, sub_prefix):
                    folder_path = os.path.join(base_dir, f"{sub_prefix}{file_code}")
                    if not os.path.exists(folder_path): return None
                    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    return os.path.join(folder_path, files[0]) if files else None

                mix_wav_path = get_wav_path(TEST_MIX_PATH, 'mixture')
                spk_wav_path = get_wav_path(TEST_SPEECH_PATH, 'speech')
                noi_wav_path = get_wav_path(TEST_NOISE_PATH, 'noise')
                mus_wav_path = get_wav_path(TEST_MUSIC_PATH, 'music')

                if not all([mix_wav_path, spk_wav_path, noi_wav_path, mus_wav_path]):
                    continue

                def load_and_fix_len(path):
                    wav, _ = librosa.load(path, sr=16000)
                    if len(wav) > target_samples:
                        wav = wav[:target_samples]
                    elif len(wav) < target_samples:
                        wav = np.pad(wav, (0, target_samples - len(wav)), mode='constant')
                    return wav

                mix_wav = load_and_fix_len(mix_wav_path)
                spk_wav = load_and_fix_len(spk_wav_path)
                noi_wav = load_and_fix_len(noi_wav_path)
                mus_wav = load_and_fix_len(mus_wav_path)

                mix_tensor = torch.from_numpy(mix_wav).float().unsqueeze(0).to(device)


                win = torch.hamming_window(512, periodic=True).to(device)
                spec = torch.stft(
                    mix_tensor, n_fft=512, hop_length=256, win_length=512, 
                    window=win, center=True, return_complex=True
                )
                test_X1 = torch.cat([spec.real, spec.imag], dim=1) # [1, 514, T]


                z_s, z_m, z_n = net(test_X1)


                est_spk = Complex_MTASS_model.Inverse_STFT(z_s, 512, 256, 512, expected_length=target_samples).cpu()
                est_mus = Complex_MTASS_model.Inverse_STFT(z_m, 512, 256, 512, expected_length=target_samples).cpu()
                est_noi = Complex_MTASS_model.Inverse_STFT(z_n, 512, 256, 512, expected_length=target_samples).cpu()


                ref_spk = torch.from_numpy(spk_wav).float().unsqueeze(0)
                ref_mus = torch.from_numpy(mus_wav).float().unsqueeze(0)
                ref_noi = torch.from_numpy(noi_wav).float().unsqueeze(0)
                ref_mix = torch.from_numpy(mix_wav).float().unsqueeze(0)

                tasks = [('speech', est_spk, ref_spk), ('music', est_mus, ref_mus), ('noise', est_noi, ref_noi)]
                for name, est, ref in tasks:
                    cur_sdr = Complex_MTASS_model.compute_sdr(ref, est)
                    cur_sisdr = Complex_MTASS_model.compute_sisdr(ref, est)
                    mix_sdr = Complex_MTASS_model.compute_sdr(ref, ref_mix)
                    mix_sisdr = Complex_MTASS_model.compute_sisdr(ref, ref_mix)

                    metrics[name]['sdr'].extend(cur_sdr.numpy())
                    metrics[name]['sdri'].extend((cur_sdr - mix_sdr).numpy())
                    metrics[name]['sisdr'].extend(cur_sisdr.numpy())
                    metrics[name]['sisdri'].extend((cur_sisdr - mix_sisdr).numpy())
                if save_count < num_save:
                    sample_dir = os.path.join(save_dir_base, f"sample_{file_code}")
                    os.makedirs(sample_dir, exist_ok=True)
                    for wav_t, name in [(est_spk, "est_speech"), (est_mus, "est_music"), (est_noi, "est_noise"), (ref_mix, "mix")]:
                        wav_np = wav_t.numpy().reshape((-1, 1))
                        if np.max(np.abs(wav_np)) > 0:
                            wav_np = (wav_np / np.max(np.abs(wav_np))) * 0.9
                        wav_write(wav_np, sample_dir, f"{name}.wav", 16000)
                    save_count += 1
        report_lines = [] 
        
        report_lines.append("\n" + "="*65)
        report_lines.append(f"{'Category':<12} | {'SDR':<7} | {'SDRi':<7} | {'SI-SDR':<7} | {'SI-SDRi':<7}")
        report_lines.append("-" * 65)
        
        totals = {'sdr': [], 'sdri': [], 'sisdr': [], 'sisdri': []}
        for name in ['speech', 'music', 'noise']:
            res = {k: np.mean(v) for k, v in metrics[name].items()}
            line = f"{name.capitalize():<12} | {res['sdr']:7.2f} | {res['sdri']:7.2f} | {res['sisdr']:7.2f} | {res['sisdri']:7.2f} (dB)"
            report_lines.append(line)
            for k in totals: totals[k].append(res[k])

        report_lines.append("-" * 65)
        avg_line = f"{'AVERAGE':<12} | {np.mean(totals['sdr']):7.2f} | {np.mean(totals['sdri']):7.2f} | {np.mean(totals['sisdr']):7.2f} | {np.mean(totals['sisdri']):7.2f} (dB)"
        report_lines.append(avg_line)
        report_lines.append("="*65)
        report_lines.append(f"音频样本已保存至: {os.path.abspath(save_dir_base)}")

        final_report_str = "\n".join(report_lines)
        
        print(final_report_str)

        report_file_path = os.path.join(save_dir_base, "evaluation_report.txt")
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write(final_report_str)
            
        print(f"评估报告文件已成功保存至: {report_file_path}")



# ---------------------------------------------------------------------------------------
# * Class:
#     MyDataset()---Make datasets for dataloader
#
#   * Arguments:
#    * datapath -- The data path of .npy file
#
#   * Returns:
#    * X1 -- Mixture RI
#    * Y1 -- Speech RI
#    * Y2 -- Noise RI
#    * Y3 -- Music RI
#
# -----------------------------------------------------------------------------------------

class MyDataset(Dataset):
    def __init__(self, input_path, target1_path, target2_path, target3_path, target4_path, target5_path, target6_path):
        # Loading .npy data in threads        
        self.X1 = np.load(input_path)
        self.Y1 = np.load(target1_path)
        self.Y2 = np.load(target2_path)
        self.Y3 = np.load(target3_path)
        self.Y4 = np.load(target4_path)
        self.Y5 = np.load(target5_path)
        self.Y6 = np.load(target6_path) 
    
    def __getitem__(self, index):
        # Mixture RI, self.X1.shape=[?,514,re_sen_len]
        # Speech RI, self.Y1.shape=[?,514,re_sen_len]
        # Noise RI, self.Y2.shape=[?,514,re_sen_len]
        # Music RI, self.Y3.shape=[?,514,re_sen_len]
        # Speech sentence, self.Y4.shape=[?,256*(re_sen_len-1)]
        # Noise sentence, self.Y5.shape=[?,256*(re_sen_len-1)]
        # Music sentence, self.Y6.shape=[?,256*(re_sen_len-1)]
                                
        X1 = self.X1[index,:,:] # X1.shape=[514,sen_len]
        Y1 = self.Y1[index,:,:] # Y1.shape=[514,sen_len]
        Y2 = self.Y2[index,:,:] # Y2.shape=[514,sen_len]
        Y3 = self.Y3[index,:,:] # Y3.shape=[514,sen_len]
        Y4 = self.Y4[index,:] # Y4.shape=[256*(re_sen_len-1)]
        Y5 = self.Y5[index,:] # Y5.shape=[256*(re_sen_len-1)]
        Y6 = self.Y6[index,:] # Y6.shape=[256*(re_sen_len-1)]
        
        X1 = torch.from_numpy(X1).float()
        Y1 = torch.from_numpy(Y1).float()
        Y2 = torch.from_numpy(Y2).float()
        Y3 = torch.from_numpy(Y3).float()
        Y4 = torch.from_numpy(Y4).float()
        Y5 = torch.from_numpy(Y5).float()
        Y6 = torch.from_numpy(Y6).float()
        return X1,Y1,Y2,Y3,Y4,Y5,Y6
        
    def __len__(self):
        return self.X1.shape[0] # The total number of audio sentences






















