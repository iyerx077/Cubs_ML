from urllib.request import urlopen
from bs4 import BeautifulSoup
from pybaseball import batting_stats
import tensorflow as tf
from pybaseball import parks
from pybaseball import playerid_lookup
import pandas as pd

batting_stats(2025)
PCA_Stats = multi_season_data[multi_season_data['Name'] == 'Pet']
url = "https://www.fangraphs.com/players/pete-crow-armstrong/27769/stats/batting"
page = urlopen(url)
html = page.read().decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
print (soup)

