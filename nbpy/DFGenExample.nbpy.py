

"""[code]"""
from importlib import reload
import os
import dfgen as dfg
reload(dfg)
""""""


"""[code]"""
DATA_ROOT=f'{os.environ["DATA"]}/planet'
DATA_ROOT
""""""


"""[code]"""
TAGS=[
    'primary',
    'clear',
    'agriculture',
    'road',
    'water',
    'partly_cloudy',
    'cultivation',
    'habitation',
    'haze',
    'cloudy',
    'bare_ground',
    'selective_logging',
    'artisinal_mine',
    'blooming',
    'slash_burn',
    'conventional_mine',
    'blow_down']
""""""


"""[code]"""
gen=dfg.DFGen(
    csv_file=f'{DATA_ROOT}/train.csv',
    image_column='image_name',
    label_column='labels',
    tags=TAGS,
    tags_to_labels_column='tags',
    image_ext='tif',
    lambda_func=False,
    csv_sep=',')
""""""


"""[code]"""
gen.dataframe.head()
""""""


"""[code]"""
gen.size
""""""


"""[code]"""
# get largest dataset with 40% blowdown
gen.require_label('blow_down',40)
""""""


"""[code]"""
gen.dataframe.head(10)
""""""


"""[code]"""
# get largest dataset with 40% blowdown
# but change all other tags to "other" category
gen=dfg.DFGen(
    csv_file=f'{DATA_ROOT}/train.csv',
    image_column='image_name',
    label_column='labels',
    tags=TAGS,
    tags_to_labels_column='tags',
    image_ext='tif',
    lambda_func=False,
    csv_sep=',')
gen.require_label('blow_down',40,reduce_to_others=True)
""""""


"""[code]"""
gen.require_label('blow_down',40)
""""""


"""[code]"""
gen.dataframe.head(10)
""""""


"""[markdown]
---
### LOAD FROM CONFIG
"""


"""[code]"""
gen2=dfg.DFGen(
    csv_file=f'{DATA_ROOT}/train.csv',
    image_ext='tif',
    lambda_func=False,
    csv_sep=',')
""""""


"""[code]"""
gen.dataframe.head(10)
""""""


