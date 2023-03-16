# Stress analysis

import pandas as pd
import numpy as np

path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Testset.csv'
df = pd.read_table(path, delimiter=";")
# print(df['Time'])

# Remove last rows where time = zero and for now; remove the rows where head position is 0
dataframe_ = df[df.Time != 0.00000]
dataframe = dataframe_[dataframe_.HeadPosition_X != 0.00000]

# print(dataframe['HeadRotation_X'].head())

# Add columns to dataframe that center values around 360 or 0
df2 = dataframe.assign(HeadRotation_X_wrap=np.unwrap(dataframe['HeadRotation_X'], period=360), HeadRotation_Y_wrap=np.unwrap(dataframe['HeadRotation_Y'], period=360), HeadRotation_Z_wrap=np.unwrap(dataframe['HeadRotation_Z'], period=360))

# Calculate standard deviations of positions x, y, z and rotation x, y and z
# print(np.std(df2['HeadPosition_X']))
# print(np.std(df2['HeadPosition_Y']))
# print(np.std(df2['HeadPosition_Z']))

# print(np.std(df2['HeadRotation_X_wrap']))
# print(np.std(df2['HeadRotation_Y_wrap']))
# print(np.std(df2['HeadRotation_Z_wrap']))

