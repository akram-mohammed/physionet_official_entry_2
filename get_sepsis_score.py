#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import os, shutil, zipfile
from numpy import array
import csv
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from scipy.stats import entropy
import scipy as sc
from zipfile import ZipFile
from sklearn.externals import joblib

def load_sepsis_model():
	# Load the saved model pickle file
	Trained_model = joblib.load('saved_model.pkl')
	return Trained_model

def get_sepsis_score(data1, Trained_model):
	#Testing
	df_test = data1
	#Forward fill missing values
	df_test.fillna(method='ffill', axis=0, inplace=True)
	df_test = pd.DataFrame(df_test).fillna(0)
	#count = 0
	df_test['ID'] = 0
	DBP = pd.pivot_table(df_test,values='DBP',index='ID',columns='ICULOS')
	O2Sat = pd.pivot_table(df_test,values='O2Sat',index='ID',columns='ICULOS')
	Temp = pd.pivot_table(df_test,values='Temp',index='ID',columns='ICULOS')
	RR = pd.pivot_table(df_test,values='Resp',index='ID',columns='ICULOS')
	BP = pd.pivot_table(df_test,values='SBP',index='ID',columns='ICULOS')
	latest = pd.pivot_table(df_test,values='HR',index='ID',columns='ICULOS')
	Heart_rate_test = latest 
	RR_test = RR 
	BP_test = BP 
	DBP_test = DBP 
	Temp_test = Temp 
	O2Sat_test = O2Sat 

	result = Heart_rate_test

	result = result.fillna(0)
	RR_test = RR_test.fillna(0)
	BP_test = BP_test.fillna(0)
	Temp_test = Temp_test.fillna(0)
	DBP_test = DBP_test.fillna(0)
	O2Sat_test = O2Sat_test.fillna(0)
	
	#Since we are using a windows-based approach (6-hour window size), we pad our output for the 6 hours following patients admission.
	scores_list = [0.9,0.9,0.9,0.9,0.9,0.9]
	labels_list = [1,1,1,1,1,1]

	scores1 = []
	labels1 = []
	#Get dataframe of probs
	#Windows based approach
	for iterat in range(0,RR_test.shape[1]-6): 
		
		for i in range (iterat,iterat+1): 
			Heart_rate_test = result.iloc[:, i:i+6]
			RR2_test = RR_test.iloc[:, i:i+6]
			BP2_test = BP_test.iloc[:, i:i+6]
			Temp2_test = Temp_test.iloc[:, i:i+6]
			DBP2_test = DBP_test.iloc[:, i:i+6]
			O2Sat2_test = O2Sat_test.iloc[:, i:i+6]

			result['HR_min'] = Heart_rate_test.min(axis=1)
			result['HR_mean'] = Heart_rate_test.mean(axis=1)
			result['HR_max'] = Heart_rate_test.max(axis=1)
			result['HR_stdev'] = Heart_rate_test.std(axis=1)
			result['HR_var'] = Heart_rate_test.var(axis=1)
			result['HR_skew'] = Heart_rate_test.skew(axis=1)
			result['HR_kurt'] = Heart_rate_test.kurt(axis=1)
			
			result['BP_min'] = BP2_test.min(axis=1)
			result['BP_mean'] = BP2_test.mean(axis=1)
			result['BP_max'] = BP2_test.max(axis=1)
			result['BP_stdev'] = BP2_test.std(axis=1)
			result['BP_var'] = BP2_test.var(axis=1)
			result['BP_skew'] = BP2_test.skew(axis=1)
			result['BP_kurt'] = BP2_test.kurt(axis=1)

			result['RR_min'] = RR2_test.min(axis=1)
			result['RR_mean'] = RR2_test.mean(axis=1)
			result['RR_max'] = RR2_test.max(axis=1)
			result['RR_stdev'] = RR2_test.std(axis=1)
			result['RR_var'] = RR2_test.var(axis=1)
			result['RR_skew'] = RR2_test.skew(axis=1)
			result['RR_kurt'] = RR2_test.kurt(axis=1)

			result['DBP_min'] = DBP2_test.min(axis=1)
			result['DBP_mean'] = DBP2_test.mean(axis=1)
			result['DBP_max'] = DBP2_test.max(axis=1)
			result['DBP_stdev'] = DBP2_test.std(axis=1)
			result['DBP_var'] = DBP2_test.var(axis=1)
			result['DBP_skew'] = DBP2_test.skew(axis=1)
			result['DBP_kurt'] = DBP2_test.kurt(axis=1)

			result['O2Sat_min'] = O2Sat2_test.min(axis=1)
			result['O2Sat_mean'] = O2Sat2_test.mean(axis=1)
			result['O2Sat_max'] = O2Sat2_test.max(axis=1)
			result['O2Sat_stdev'] = O2Sat2_test.std(axis=1)
			result['O2Sat_var'] = O2Sat2_test.var(axis=1)
			result['O2Sat_skew'] = O2Sat2_test.skew(axis=1)
			result['O2Sat_kurt'] = O2Sat2_test.kurt(axis=1)

			result['Temp_min'] = Temp2_test.min(axis=1)
			result['Temp_mean'] = Temp2_test.mean(axis=1)
			result['Temp_max'] = Temp2_test.max(axis=1)
			result['Temp_stdev'] = Temp2_test.std(axis=1)
			result['Temp_var'] = Temp2_test.var(axis=1)
			result['Temp_skew'] = Temp2_test.skew(axis=1)
			result['Temp_kurt'] = Temp2_test.kurt(axis=1)
	 
			X_test = result.values[:, Temp2_test.shape[1]:Temp2_test.shape[1]+42] 

			scores = Trained_model.predict_proba(X_test)
			scores1.append(scores[0][1])
			
			if scores1[0]>=0.3:
				labels = 1
			else:
				labels = 0
			labels1.append(labels)
	return (scores_list+scores1, labels_list+labels1)

































import numpy as np

def get_sepsis_score(data, model):
    x_mean = np.array([
        83.8996, 97.0520,  36.8055,  126.2240, 86.2907,
        66.2070, 18.7280,  33.7373,  -3.1923,  22.5352,
        0.4597,  7.3889,   39.5049,  96.8883,  103.4265,
        22.4952, 87.5214,  7.7210,   106.1982, 1.5961,
        0.6943,  131.5327, 2.0262,   2.0509,   3.5130,
        4.0541,  1.3423,   5.2734,   32.1134,  10.5383,
        38.9974, 10.5585,  286.5404, 198.6777])
    x_std = np.array([
        17.6494, 3.0163,  0.6895,   24.2988, 16.6459,
        14.0771, 4.7035,  11.0158,  3.7845,  3.1567,
        6.2684,  0.0710,  9.1087,   3.3971,  430.3638,
        19.0690, 81.7152, 2.3992,   4.9761,  2.0648,
        1.9926,  45.4816, 1.6008,   0.3793,  1.3092,
        0.5844,  2.5511,  20.4142,  6.4362,  2.2302,
        29.8928, 7.0606,  137.3886, 96.8997])
    c_mean = np.array([60.8711, 0.5435, 0.0615, 0.0727, -59.6769, 28.4551])
    c_std = np.array([16.1887, 0.4981, 0.7968, 0.8029, 160.8846, 29.5367])

    x = data[-1, 0:34]
    c = data[-1, 34:40]
    x_norm = np.nan_to_num((x - x_mean) / x_std)
    c_norm = np.nan_to_num((c - c_mean) / c_std)

    beta = np.array([
        0.1806,  0.0249, 0.2120,  -0.0495, 0.0084,
        -0.0980, 0.0774, -0.0350, -0.0948, 0.1169,
        0.7476,  0.0323, 0.0305,  -0.0251, 0.0330,
        0.1424,  0.0324, -0.1450, -0.0594, 0.0085,
        -0.0501, 0.0265, 0.0794,  -0.0107, 0.0225,
        0.0040,  0.0799, -0.0287, 0.0531,  -0.0728,
        0.0243,  0.1017, 0.0662,  -0.0074, 0.0281,
        0.0078,  0.0593, -0.2046, -0.0167, 0.1239])
    rho = 7.8521
    nu = 1.0389

    xstar = np.concatenate((x_norm, c_norm))
    exp_bx = np.exp(np.dot(xstar, beta))
    l_exp_bx = pow(4 / rho, nu) * exp_bx

    score = 1 - np.exp(-l_exp_bx)
    label = score > 0.45

    return score, label

def load_sepsis_model():
    return None
