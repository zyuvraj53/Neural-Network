import numpy as np
import pandas as pd

weather = pd.read_csv("weather.csv", index_col="DATE")
print(weather)