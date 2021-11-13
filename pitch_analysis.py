from mplsoccer.statsbomb import read_event, EVENT_SLUG
from mplsoccer import Pitch, VerticalPitch
from mplsoccer.cm import create_transparent_cmap
from mplsoccer.scatterutils import arrowhead_marker
from mplsoccer.statsbomb import read_event, EVENT_SLUG

from mplsoccer.utils import FontManager
import matplotlib.pyplot as plt
from statsbombpy import sb
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

world_cup_games = sb.matches(competition_id=43, season_id=3)
world_cup_games['game'] = world_cup_games['home_team'] + ' vs ' + world_cup_games['away_team']


kwargs = {'related_event_df': False, 'shot_freeze_frame_df': False,
          'tactics_lineup_df': False, 'warn': False}
df = read_event(f'{EVENT_SLUG}/7580.json', **kwargs)['event']

# subset the barcelona shots
df_shots = df[(df.type_name == 'Shot') & (df.team_name == 'France')].copy()
# subset the barca open play passes
df_pass = df[(df.type_name == 'Pass') &
                   (df.team_name == 'France') &
                   (~df.sub_type_name.isin(['Throw-in', 'Corner', 'Free Kick', 'Kick Off']))].copy()

# setup a mplsoccer FontManager to download google fonts (Roboto-Regular / SigmarOne-Regular)
fm_rubik = FontManager(('https://github.com/google/fonts/blob/main/ofl/rubikmonoone/'
                        'RubikMonoOne-Regular.ttf?raw=true'))