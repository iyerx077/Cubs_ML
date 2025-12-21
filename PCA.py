from urllib.request import urlopen
from bs4 import BeautifulSoup
from pybaseball import batting_stats
import tensorflow as tf
from pybaseball import parks
from pybaseball import playerid_lookup
from pybaseball import statcast_batter
from matplotlib import pyplot
import pandas as pd
import numpy as np
import scipy
import math

import geneNewData
from precode import *

#using pybaseball to explore Pete Crow-Armstrong's performance in the 2025 season
data = batting_stats(2025)
PCA = playerid_lookup("Crow-Armstrong")
print ("Player: ", PCA['key_mlbam'].values)
'''
Data:  Index(['IDfg', 'Season', 'Name', 'Team', 'Age', 'G', 'AB', 'PA', 'H', '1B',
       ...
       'maxEV', 'HardHit', 'HardHit%', 'Events', 'CStr%', 'CSW%', 'xBA',
       'xSLG', 'xwOBA', 'L-WAR'],'''
PCA_stats = statcast_batter('2025-03-18','2025-10-11',PCA['key_mlbam'].values[0])
#pyplot.plot(PCA_stats)
'''Index(['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
       'release_pos_z', 'player_name', 'batter', 'pitcher', 'events',
       'description',
       ...
       'batter_days_until_next_game', 'api_break_z_with_gravity',
       'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'attack_angle',
       'attack_direction', 'swing_path_tilt',
       'intercept_ball_minus_batter_pos_x_inches',
       'intercept_ball_minus_batter_pos_y_inches'],'''
pitch_type_dict = PCA_stats['pitch_type'].to_dict()
pitch_type_description = {}
for i in pitch_type_dict.keys():
    j = PCA_stats['description'][i]
    tuple = (pitch_type_dict[i], j)
    if tuple not in pitch_type_description.keys():
        pitch_type_description[tuple] = 0
    else:
        pitch_type_description[tuple] += 1

sub_train_images, sub_train_labels, sub_test_images, sub_test_labels = init_subset('1670')
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
print (model.variables)
print (pitch_type_description)
print (len(pitch_type_description))
#print (data['Team'])

#Density Estimation and Classification
def main():
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
    return e / denominator

