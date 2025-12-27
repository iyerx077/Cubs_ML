from urllib.request import urlopen
from bs4 import BeautifulSoup
from pybaseball import batting_stats
import tensorflow as tf
from pybaseball import parks
from pybaseball import playerid_lookup
from pybaseball import statcast_batter
from pybaseball import batting_stats
from matplotlib import pyplot
import pandas as pd
import numpy as np
import scipy
import math

import geneNewData
from precode import *

#using pybaseball to explore Pete Crow-Armstrong's performance in the 2025 season
PCA = playerid_lookup("Crow-Armstrong")
#Index(['name_last', 'name_first', 'key_mlbam', 'key_retro', 'key_bbref','key_fangraphs', 'mlb_played_first', 'mlb_played_last'],dtype='object')
#['IDfg', 'Season', 'Name', 'Team', 'Age', 'G', 'AB', 'PA', 'H', '1B', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 'SF', 'SH', 'GDP', 'SB', 'CS', 'AVG', 'GB', 'FB', 'LD', 'IFFB', 'Pitches', 'Balls', 'Strikes', 'IFH', 'BU', 'BUH', 'BB%', 'K%', 'BB/K', 'OBP', 'SLG', 'OPS', 'ISO', 'BABIP', 'GB/FB', 'LD%', 'GB%', 'FB%', 'IFFB%', 'HR/FB', 'IFH%', 'BUH%', 'wOBA', 'wRAA', 'wRC', 'Bat', 'Fld', 'Rep', 'Pos', 'RAR', 'WAR', 'Dol', 'Spd', 'wRC+', 'WPA', '-WPA', '+WPA', 'RE24', 'REW', 'pLI', 'phLI', 'PH', 'WPA/LI', 'Clutch', 'FB% (Pitch)', 'FBv', 'SL%', 'SLv', 'CT%', 'CTv', 'CB%', 'CBv', 'CH%', 'CHv', 'SF%', 'SFv', 'KN%', 'KNv', 'XX%', 'PO%', 'wFB', 'wSL', 'wCT', 'wCB', 'wCH', 'wSF', 'wKN', 'wFB/C', 'wSL/C', 'wCT/C', 'wCB/C', 'wCH/C', 'wSF/C', 'wKN/C', 'O-Swing%', 'Z-Swing%', 'Swing%', 'O-Contact%', 'Z-Contact%', 'Contact%', 'Zone%', 'F-Strike%', 'SwStr%', 'BsR', 'FA% (sc)', 'FT% (sc)', 'FC% (sc)', 'FS% (sc)', 'FO% (sc)', 'SI% (sc)', 'SL% (sc)', 'CU% (sc)', 'KC% (sc)', 'EP% (sc)', 'CH% (sc)', 'SC% (sc)', 'KN% (sc)', 'UN% (sc)', 'vFA (sc)', 'vFT (sc)', 'vFC (sc)', 'vFS (sc)', 'vFO (sc)', 'vSI (sc)', 'vSL (sc)', 'vCU (sc)', 'vKC (sc)', 'vEP (sc)', 'vCH (sc)', 'vSC (sc)', 'vKN (sc)', 'FA-X (sc)', 'FT-X (sc)', 'FC-X (sc)', 'FS-X (sc)', 'FO-X (sc)', 'SI-X (sc)', 'SL-X (sc)', 'CU-X (sc)', 'KC-X (sc)', 'EP-X (sc)', 'CH-X (sc)', 'SC-X (sc)', 'KN-X (sc)', 'FA-Z (sc)', 'FT-Z (sc)', 'FC-Z (sc)', 'FS-Z (sc)', 'FO-Z (sc)', 'SI-Z (sc)', 'SL-Z (sc)', 'CU-Z (sc)', 'KC-Z (sc)', 'EP-Z (sc)', 'CH-Z (sc)', 'SC-Z (sc)', 'KN-Z (sc)', 'wFA (sc)', 'wFT (sc)', 'wFC (sc)', 'wFS (sc)', 'wFO (sc)', 'wSI (sc)', 'wSL (sc)', 'wCU (sc)', 'wKC (sc)', 'wEP (sc)', 'wCH (sc)', 'wSC (sc)', 'wKN (sc)', 'wFA/C (sc)', 'wFT/C (sc)', 'wFC/C (sc)', 'wFS/C (sc)', 'wFO/C (sc)', 'wSI/C (sc)', 'wSL/C (sc)', 'wCU/C (sc)', 'wKC/C (sc)', 'wEP/C (sc)', 'wCH/C (sc)', 'wSC/C (sc)', 'wKN/C (sc)', 'O-Swing% (sc)', 'Z-Swing% (sc)', 'Swing% (sc)', 'O-Contact% (sc)', 'Z-Contact% (sc)', 'Contact% (sc)', 'Zone% (sc)', 'Pace', 'Def', 'wSB', 'UBR', 'Age Rng', 'Off', 'Lg', 'wGDP', 'Pull%', 'Cent%', 'Oppo%', 'Soft%', 'Med%', 'Hard%', 'TTO%', 'CH% (pi)', 'CS% (pi)', 'CU% (pi)', 'FA% (pi)', 'FC% (pi)', 'FS% (pi)', 'KN% (pi)', 'SB% (pi)', 'SI% (pi)', 'SL% (pi)', 'XX% (pi)', 'vCH (pi)', 'vCS (pi)', 'vCU (pi)', 'vFA (pi)', 'vFC (pi)', 'vFS (pi)', 'vKN (pi)', 'vSB (pi)', 'vSI (pi)', 'vSL (pi)', 'vXX (pi)', 'CH-X (pi)', 'CS-X (pi)', 'CU-X (pi)', 'FA-X (pi)', 'FC-X (pi)', 'FS-X (pi)', 'KN-X (pi)', 'SB-X (pi)', 'SI-X (pi)', 'SL-X (pi)', 'XX-X (pi)', 'CH-Z (pi)', 'CS-Z (pi)', 'CU-Z (pi)', 'FA-Z (pi)', 'FC-Z (pi)', 'FS-Z (pi)', 'KN-Z (pi)', 'SB-Z (pi)', 'SI-Z (pi)', 'SL-Z (pi)', 'XX-Z (pi)', 'wCH (pi)', 'wCS (pi)', 'wCU (pi)', 'wFA (pi)', 'wFC (pi)', 'wFS (pi)', 'wKN (pi)', 'wSB (pi)', 'wSI (pi)', 'wSL (pi)', 'wXX (pi)', 'wCH/C (pi)', 'wCS/C (pi)', 'wCU/C (pi)', 'wFA/C (pi)', 'wFC/C (pi)', 'wFS/C (pi)', 'wKN/C (pi)', 'wSB/C (pi)', 'wSI/C (pi)', 'wSL/C (pi)', 'wXX/C (pi)', 'O-Swing% (pi)', 'Z-Swing% (pi)', 'Swing% (pi)', 'O-Contact% (pi)', 'Z-Contact% (pi)', 'Contact% (pi)', 'Zone% (pi)', 'Pace (pi)', 'FRM', 'AVG+', 'BB%+', 'K%+', 'OBP+', 'SLG+', 'ISO+', 'BABIP+', 'LD+%', 'GB%+', 'FB%+', 'HR/FB%+', 'Pull%+', 'Cent%+', 'Oppo%+', 'Soft%+', 'Med%+', 'Hard%+', 'EV', 'LA', 'Barrels', 'Barrel%', 'maxEV', 'HardHit', 'HardHit%', 'Events', 'CStr%', 'CSW%', 'xBA', 'xSLG', 'xwOBA', 'L-WAR']
Stats = batting_stats(PCA['mlb_played_first'],PCA['mlb_played_last'])
PCA_Stat_Cast = statcast_batter('2025-03-18','2025-10-11',PCA['key_mlbam'].values[0])
#PCA_Stat_Cast.to_csv("PCA_Stat_Cast.csv")
PCA_events = PCA_Stat_Cast.groupby(['events','pitch_type'])
#['pitch_type', 'game_date', 'release_speed', 'release_pos_x', 'release_pos_z', 'player_name', 'batter', 'pitcher', 'events', 'description', 'spin_dir', 'spin_rate_deprecated', 'break_angle_deprecated', 'break_length_deprecated', 'zone', 'des', 'game_type', 'stand', 'p_throws', 'home_team', 'away_team', 'type', 'hit_location', 'bb_type', 'balls', 'strikes', 'game_year', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot', 'hc_x', 'hc_y', 'tfs_deprecated', 'tfs_zulu_deprecated', 'umpire', 'sv_id', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'hit_distance_sc', 'launch_speed', 'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 'game_pk', 'fielder_2', 'fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9', 'release_pos_y', 'estimated_ba_using_speedangle', 'estimated_woba_using_speedangle', 'woba_value', 'woba_denom', 'babip_value', 'iso_value', 'launch_speed_angle', 'at_bat_number', 'pitch_number', 'pitch_name', 'home_score', 'away_score', 'bat_score', 'fld_score', 'post_away_score', 'post_home_score', 'post_bat_score', 'post_fld_score', 'if_fielding_alignment', 'of_fielding_alignment', 'spin_axis', 'delta_home_win_exp', 'delta_run_exp', 'bat_speed', 'swing_length', 'estimated_slg_using_speedangle', 'delta_pitcher_run_exp', 'hyper_speed', 'home_score_diff', 'bat_score_diff', 'home_win_exp', 'bat_win_exp', 'age_pit_legacy', 'age_bat_legacy', 'age_pit', 'age_bat', 'n_thruorder_pitcher', 'n_priorpa_thisgame_player_at_bat', 'pitcher_days_since_prev_game', 'batter_days_since_prev_game', 'pitcher_days_until_next_game', 'batter_days_until_next_game', 'api_break_z_with_gravity', 'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'attack_angle', 'attack_direction', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 'intercept_ball_minus_batter_pos_y_inches']
#pyplot.plot(PCA_Stat_Cast)
pitch_type_dict = PCA_Stat_Cast['pitch_type'].to_dict()
#arm_angle = PCA_Stat_Cast['arm_angle'].to_dict()
pitch_type_description = {}
pitch_type_event = {}
for i in pitch_type_dict.keys():
    j = PCA_Stat_Cast['description'][i]
    k = PCA_Stat_Cast['event'][i]
    tuple = (pitch_type_dict[i], j)
    if tuple not in pitch_type_description.keys():
        pitch_type_description[tuple] = 0
    else:
        pitch_type_description[tuple] += 1

'''sub_train_images, sub_train_labels, sub_test_images, sub_test_labels = init_subset('1670')
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
print (model.variables)'''
print (pitch_type_description)
print (len(pitch_type_description))
#print (data['Team'])

#Density Estimation and Classification
'''def main():
    myID = '1670'  # change to last 4 digit of your studentID
    geneNewData.geneData(myID)
    Numpyfile0 = scipy.io.loadmat('digit0_stu_train' + myID + '.mat')
    Numpyfile1 = scipy.io.loadmat('digit1_stu_train' + myID + '.mat')
    Numpyfile2 = scipy.io.loadmat('digit0_testset' + '.mat')
    Numpyfile3 = scipy.io.loadmat('digit1_testset' + '.mat')
    train0 = Numpyfile0.get('target_img')
    train1 = Numpyfile1.get('target_img')
    test0 = Numpyfile2.get('target_img')
    test1 = Numpyfile3.get('target_img')

    # Task 1
    train0set = []
    train1set = []

    # extracting datapoints into 2d arrays as [mean,variance]
    for i in range(len(train0)):  # for both tests since train0 and train1 have the same lenght
        train0set.append([np.mean(train0[i]), np.std(train0[i])])
        train1set.append([np.mean(train1[i]), np.std(train1[i])])

    # Task 2
    # using axis=0 since it will calculate the same indexed elements for all of the arrays
    meantrain0 = np.mean(train0set, axis=0)
    vartrain0 = np.var(train0set, axis=0)
    meantrain1 = np.mean(train1set, axis=0)
    vartrain1 = np.var(train1set, axis=0)

    # assigning the mean and variances of the training sets to variables
    mf1train0 = meantrain0[0]
    vf1train0 = vartrain0[0]
    mf2train0 = meantrain0[1]
    vf2train0 = vartrain0[1]
    mf1train1 = meantrain1[0]
    vf1train1 = vartrain1[0]
    mf2train1 = meantrain1[1]
    vf2train1 = vartrain1[1]

    # Task 3
    test0set = []
    test1set = []

    # extracting datapoints into 2d arrays as [mean,standard deviation]
    for j in range(len(test0)):  # for both tests since train0 and train1 have the same lenght
        test0set.append([np.mean(test0[j]), np.std(test0[j])])

    for t in range(len(test1)):
        test1set.append([np.mean(test1[t]), np.std(test1[t])])

    samplespredicted1 = 0
    samplespredicted0 = 0

    # compute P(X|y=0) by multiplying the PDF with the input test datapoints for std and mean and probability of each digit
    for f in range(len(test0set)):
        probf1train0 = NB(test0set[f][0], mf1train0, math.sqrt(vf1train0))
        probf2train0 = NB(test0set[f][1], mf2train0, math.sqrt(vf2train0))
        expvaluetrain0 = probf1train0 * probf2train0 * 0.5

        probf1train1 = NB(test0set[f][0], mf1train1, math.sqrt(vf1train1))
        probf2train1 = NB(test0set[f][1], mf2train1, math.sqrt(vf2train1))
        expvaluetrain1 = probf1train1 * probf2train1 * 0.5

        # e

        if expvaluetrain0 > expvaluetrain1:
            samplespredicted0 += 1

    # compute P(X|y=1) by multiplying the PDF with the input test datapoints for std and mean and probability of each digit
    for k in range(len(test1set)):
        probf1train0 = NB(test1set[k][0], mf1train0, math.sqrt(vf1train0))
        probf2train0 = NB(test1set[k][1], mf2train0, math.sqrt(vf2train0))
        expectedprobtrain0 = probf1train0 * probf2train0 * 0.5

        probf1train1 = NB(test1set[k][0], mf1train1, math.sqrt(vf1train1))
        probf2train1 = NB(test1set[k][1], mf2train1, math.sqrt(vf2train1))
        expectedprobtrain1 = probf1train1 * probf2train1 * 0.5

        if expectedprobtrain0 < expectedprobtrain1:
            samplespredicted1 += 1

    # Task 4
    # Calculating the accuracy by dividing the number of correctly predicted labels by the number of samples
    Accuracy_for_digit0testset = samplespredicted0 / len(test0set)
    Accuracy_for_digit1testset = samplespredicted1 / len(test1set)

    # implement NB calssifiers parameters from task 2; use classifiers to predict unknown labels

    print(['1670', mf1train0, vf1train0, mf2train0, vf2train0, mf1train1, vf1train1, mf2train1, vf2train1,
           Accuracy_for_digit0testset, Accuracy_for_digit1testset])
    # ['ASUId', Mean_of_feature1_for_digit0, Variance_of_feature1_for_digit0, Mean_of_feature2_for_digit0, Variance_of_feature2_for_digit0 , Mean_of_feature1_for_digit1, Variance_of_feature1_for_digit1, Mean_of_feature2_for_digit1, Variance_of_feature2_for_digit1, Accuracy_for_digit0testset, Accuracy_for_digit1testset]

    # each of the matrices in the train0 and train1 has 28 elements

    # print([len(train0),len(train1),len(test0),len(test1)])

    # print('Your trainset and testset are generated successfully!')
    pass


# NB classifier formula
def NB(test, avg, omega):
    exponent = -0.5 * (((test - avg) / omega) ** 2)
    e = math.e ** (exponent)
    denominator = omega * math.sqrt(2 * math.pi)
    return e / denominator'''

