#!/usr/bin/env python
# coding: utf-8

import shutil
import sys
import glob
import tarfile
import os
import mne
import pandas as pd
import numpy as np
from tqdm import tqdm
from directory_tree import display_tree
from mne_connectivity import spectral_connectivity_epochs
from mne.datasets import sample
import numpy as np
from collections import Counter
import ast

def ReadFile():
    EEG = []
    MetaData = pd.read_csv('/workspace/brail_poc3/experiments/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    
    # file list over 40
    fileList = [item for item in sorted(glob.glob("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/*")) if item.split('/')[-1] in ground_truth["ID"].tolist()]
    # file list all
#     fileList = sorted(glob.glob("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/*"))
    
    for path in tqdm(fileList):
        print("Subject number:", path.split('/')[-1])
        Subjects = {}
        Subjects["Subject_Number"] = path.split('/')[-1]
        Subjects["prepSignal_EC"] = mne.io.read_raw_eeglab("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/{}/{}/{}_EC.set".format(Subjects["Subject_Number"], Subjects["Subject_Number"], Subjects["Subject_Number"]), preload=True, verbose=False)
        Subjects["prepSignal_EC"] = InterpolateCh(Subjects["Subject_Number"], Subjects["prepSignal_EC"])
        
#         Subjects["prepSignal_EO"] = mne.io.read_raw_eeglab("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/{}/{}/{}_EO.set".format(Subjects["Subject_Number"], Subjects["Subject_Number"], Subjects["Subject_Number"]), preload=True, verbose=False)
#         Subjects["prepSignal_EO"] = InterpolateCh(Subjects["Subject_Number"], Subjects["prepSignal_EO"])
        
        Subjects["Age"] = MetaData[MetaData['Subject'] == Subjects["Subject_Number"]].iloc[0,2]
        Subjects["Age_Number"] = int(Subjects["Age"].split('-')[0])
        EEG.append(Subjects)
    return EEG

def CauConn(ChanNum, data, metrics):
    len_channel = data.shape[1]
    index1 = [ChanNum] * (len_channel-1)
    index2 = list(np.arange(0, len_channel, 1))
    del index2[ChanNum]
    indices = (index1, index2)
#     print(indices)
    conn = spectral_connectivity_epochs(
            data, method=metrics, sfreq=250, indices=indices,
            fmin=9, fmax=11, faverage=True, verbose=False).get_data()[:, 0]
    conn = conn.tolist()
    return conn

def splitEEG(prepEEG):
    epochs = []
    ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
    annotation_EC_df = prepEEG.annotations.to_data_frame()
    EC_sec = annotation_EC_df[annotation_EC_df.columns[0]].dt.strftime('%s.%f').astype(float) + 32400
    point_indices = annotation_EC_df.index[annotation_EC_df['description'] == 'boundary']
    BreakPoint = []
#     print(EC_sec[point_indices])
    for i, j in enumerate(EC_sec[point_indices]):
        if i == 0:
            BreakPoint.append([0, int(j)])
        else:
            BreakPoint.append([BreakPoint[-1][1], int(j)])
        if i == len(EC_sec[point_indices])-1:
            BreakPoint.append([int(j)+1, int(EC_sec.iloc[-1])])
#     print(BreakPoint)
    for i,d  in enumerate(BreakPoint): # Ignore the short duration between each braek point.
        if d[1] - d[0] < 15:
            del BreakPoint[i]
    for i, d in enumerate(BreakPoint):  # Splite each epoch by break point
#         print("The {} trial from {} to {}. ({}sec)".format(ordinal(i+1), d[0], d[1], d[1]-d[0]))
        events = mne.make_fixed_length_events(prepEEG.copy().crop(tmin=d[0], tmax=d[1]), duration=2, overlap=0.2)
        epochs.append(mne.Epochs(prepEEG, events=events, event_id=None, tmin=0, tmax=2, baseline=None, preload=False, verbose=False))
#         epochs.append(prepEEG.copy().crop(tmin=d[0], tmax=d[1]))
#         print("------------------------------------------------")
    return epochs

def EpochRotate(epoches, metric):
    PLIs = []
    for OneEpoch in epoches:
        Chan_PLI = []
        data = OneEpoch.get_data()
        for i in range(data.shape[1]):
            Chan_PLI.append(CauConn(i, data, metric))
        for i in range(data.shape[1]):
            Chan_PLI[i].insert(i, 0)
        PLIs.append(np.array(Chan_PLI))
    return np.array(PLIs)

def InsertCh(Ch_name, raw):
    # Need NoneDel_prepSignal to get local info
    # Used for InterpolateCh
    
    prep_len = raw._data.shape[1]
#     new_channel_chs = raw.info['chs'][-1]
#     new_channel_chs['ch_name'] = Ch_name
#     new_channel_chs['scanno'] = new_channel_chs['scanno']+1
#     new_channel_chs['logno'] = new_channel_chs['logno']+1
#     new_channel_chs['loc'] = [item.get('loc') for item in NoneDel_prepSignal.info['chs'] if(item['ch_name'] == Ch_name)][0]
#     print(new_channel_chs)
    
    new_channel_data = np.array([0] * prep_len)
    new_channel_info = mne.create_info(ch_names=[Ch_name], sfreq=raw.info['sfreq'], ch_types=['eeg'])
    new_channel_info['chs'][0]['loc'] = [item.get('loc') for item in NoneDel_prepSignal.info['chs'] if(item['ch_name'] == Ch_name)][0]
    print(new_channel_info['chs'])
    new_channel_data = new_channel_data[np.newaxis, :]  # 新しいチャンネルデータを適切な形式に変換
    new_raw = mne.io.RawArray(new_channel_data, new_channel_info)
    raw.add_channels([new_raw], force_update_info=True)
    return raw

def InterpolateCh(ID, raw):
    # Target EC, Need del_ch file
    # Interpolate missing channel in preprocessed LEMON EEG data
    
    CH = del_ch[del_ch["ID"] == ID]['del_channel']
    del_Ch_list = ast.literal_eval(CH.tolist()[0])
    prepSignal_EC_inte = raw.copy()

    for i in [item for item in del_Ch_list if item != 'VEOG']:
        prepSignal_EC_inte = InsertCh(i, prepSignal_EC_inte)
        prepSignal_EC_inte.info["bads"].append(i)

    prepSignal_EC_inte.info

    # Interpolate channels
    prepSignal_EC_inte = prepSignal_EC_inte.interpolate_bads()

    # Reorder Channel
    Ch_order = NoneDel_prepSignal.ch_names
    prepSignal_EC_inte.reorder_channels(Ch_order)
    
    return prepSignal_EC_inte

def convert_age_range(age_range):
    return int(age_range.split("-")[0])

if __name__ == "__main__":
    
    #################################
    #                               #
    # Example:                      #
    # >> python3 FC_EC_Cal.py coh   #
    #                               #
    #################################
    
    args = sys.argv
    if len(args) <= 0:
        print("Using defalut FC metric as PLI")
        metric_Name = "PLI"
    else:
        metric_Name = args[1]
        print(metric_Name)
        if not (metric_Name in ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']):
            print("Use metrics in ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased']")
            sys.exit()
            
    # Initial variables
    NoneDel_prepSignal = NoneDel_prepSignal = mne.io.read_raw_eeglab('/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/sub-032303/sub-032303/sub-032303_EC.set', preload=True)
    del_ch = pd.read_csv("/workspace/brail_poc3/notebooks/Chen/Sub_channel_list.csv", index_col=0)
    Cog_score = pd.read_csv("/workspace/brail_poc3/notebooks/Chen/TestCognitive.csv")
    Cog_score = Cog_score.loc[:, ["ID", "Label"]]
    Age_data = pd.read_csv("/workspace/brail_poc3/experiments/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")
    Age_data = Age_data.loc[:, ["Subject", "Age"]]
    Age_data = Age_data.rename({'Subject': 'ID'}, axis='columns')
    Age_data["Age_1"] = Age_data["Age"].apply(convert_age_range)
    Age_data.drop("Age", axis=1, inplace=True)
    cog_df = Cog_score.merge(Age_data, on='ID', how='left')
    ground_truth = cog_df[cog_df['Age_1'] >= 40]
    
    # Read file in dict
    print("------------------------------------------------")
    print("Read file in dict")
    EEG = ReadFile()
    print(EEG)
    # Splite EEG to epoches
    print("------------------------------------------------")
    print("Splite EEG to epoches")
    EEG_epoches = []
    for i in tqdm(range(len(EEG))):
        EEG_epoches.append(splitEEG(EEG[i]['prepSignal_EC']))
    # print(EEG_epoches)
    # Calculate EEG functional connectivity
    print("------------------------------------------------")
    print("Calculate EEG functional connectivity")
    for i, d in enumerate(tqdm(EEG_epoches)):
        FC = EpochRotate(d, metric_Name)
        np.save('/workspace/brail_poc3/experiments/FC_Result_231012/{}/{}.npy'.format(metric_Name, EEG[i]['Subject_Number']), FC)

