# Stress analysis

import pandas as pd
import numpy as np
from bisect import bisect_left


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def cut_dataframe(dataframe, person, duration_piece=10):
    # Create a dictionary for one person with the dataframe cut to pieces
    times = []
    indices = []
    d = {}
    times.append(dataframe['Time'][0])
    # Find the start and end times for each piece
    for i in range(10):
        duration_piece = 5

        if i == 0:
            time = (take_closest(list(dataframe['Time']), (dataframe['Time'][0]+duration_piece)))
            times.append(time)

        else:
            time = (take_closest(list(dataframe['Time']), time+duration_piece))
            times.append(time)

            if time - times[i-1] < 0.5*duration_piece:
                times.pop()  # remove last element from list
                break
    # Find indices of times
    for j in range(len(times)):
        ind = int(df2[df2['Time'] == times[j]].index.values)
        indices.append(ind)
    # Create a dict of the dataframes of the different pieces
    for i in range(len(indices)-1):
        d[f"dataframe{person}_{i}"] = df2.loc[indices[i]:indices[i+1]-1, :]
    return d


path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Testset.csv'
df = pd.read_table(path, delimiter=";", dtype=np.float64)
# # print(df['Time'])

# Remove last rows where time = zero and for now; remove the rows where head position is 0
dataframe_ = df[df.Time != 0.00000]
dataframe = dataframe_[dataframe_.HeadPosition_X != 0.00000]

# print(dataframe['HeadRotation_X'].head())

# # Add columns to dataframe that center values around 360 or 0
df2 = dataframe.assign(HeadRotation_X_unwrap=np.unwrap(dataframe['HeadRotation_X'], period=360), HeadRotation_Y_unwrap=np.unwrap(dataframe['HeadRotation_Y'], period=360), HeadRotation_Z_unwrap=np.unwrap(dataframe['HeadRotation_Z'], period=360))

# # Data in korte stukjes knippen
# # Vind de tijd x sec verder van vorige laatste index. loopje maken; wanneer eindigt het? als er niet meer de helft van duration_piece beschikbaar is. 
# index = []

# Dataframes van de verschillende stukjes maken
d = cut_dataframe(df2, 1)
print(d)

# # Calculate standard deviations of positions x, y, z and rotation x, y and z
# # print(np.std(df2['HeadPosition_X']))
# # print(np.std(df2['HeadPosition_Y']))
# # print(np.std(df2['HeadPosition_Z']))

# # print(np.std(df2['HeadRotation_X_unwrap']))
# # print(np.std(df2['HeadRotation_Y_unwrap']))
# # print(np.std(df2['HeadRotation_Z_unwrap']))

