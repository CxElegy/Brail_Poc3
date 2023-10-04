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

def ReadFile():
    EEG = []
    MetaData = pd.read_csv('/workspace/brail_poc3/experiments/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    for path in tqdm(sorted(glob.glob("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/*"))):
        print("Subject number:", path.split('/')[-1])
        Subjects = {}
        Subjects["Subject_Number"] = path.split('/')[-1]
        Subjects["prepSignal_EC"] = mne.io.read_raw_eeglab("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/{}/{}/{}_EC.set".format(Subjects["Subject_Number"], Subjects["Subject_Number"], Subjects["Subject_Number"]), preload=True, verbose=False)
        Subjects["prepSignal_EO"] = mne.io.read_raw_eeglab("/workspace/brail_poc3/experiments/Lemon_PreEEGLAB/{}/{}/{}_EO.set".format(Subjects["Subject_Number"], Subjects["Subject_Number"], Subjects["Subject_Number"]), preload=True, verbose=False)
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
        events = mne.make_fixed_length_events(prepEEG.copy().crop(tmin=d[0], tmax=d[1]), duration=2)
        epochs.append(mne.Epochs(prepEEG, events=events, event_id=None, tmin=0, tmax=2, baseline=None, preload=False, verbose=False))
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

if __name__ == "__main__":
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
        np.save('/workspace/brail_poc3/experiments/FC_Result/{}/{}.npy'.format(metric_Name, EEG[i]['Subject_Number']), FC)

