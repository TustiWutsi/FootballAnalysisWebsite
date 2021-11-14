import pandas as pd

from mplsoccer.statsbomb import read_event, EVENT_SLUG
from mplsoccer import Pitch, VerticalPitch
from mplsoccer.cm import create_transparent_cmap
from mplsoccer.scatterutils import arrowhead_marker
from mplsoccer.statsbomb import read_event, EVENT_SLUG

from mplsoccer.utils import FontManager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

world_cup_games = pd.read_csv('files/world_cup_games.csv')
world_cup_games['game'] = world_cup_games['home_team'] + ' vs ' + world_cup_games['away_team']