"""
This module does blah blah.

"""

import os
import pandas as pd
# import numpy as np
# import matplotlib as plt


working_dir_path = os.path.dirname(__file__)
car_price_path = os.path.join(working_dir_path, "ressources\\car_price.csv")

df = pd.read_csv(car_price_path, index_col="car_ID")
data = df[['curbweight', 'enginesize', 'highwaympg',
           'horsepower', 'citympg', 'peakrpm', 'price']]

print("========== PROGRAM COMPLETED ==========")
