from urllib.request import urlopen
from bs4 import BeautifulSoup
from pybaseball import batting_stats
import tensorflow as tf
from pybaseball import parks
from pybaseball import playerid_lookup
from pybaseball import statcast_batter
from matplotlib import pyplot
import pandas as pd

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
print (pitch_type_description)
print (len(pitch_type_description))
#print (data['Team'])

