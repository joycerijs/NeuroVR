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
    print('this is pos')
    print(pos)
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


path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Testset_nieuw.csv'
df = pd.read_table(path, delimiter=";")
# print(df['Time'])

# Remove last rows where time = zero and for now; remove the rows where head position is 0
dataframe_ = df[df.Time != 0.00000]
dataframe = dataframe_[dataframe_.HeadPosition_X != 0.00000]

# print(dataframe['HeadRotation_X'].head())

# Add columns to dataframe that center values around 360 or 0
df2 = dataframe.assign(HeadRotation_X_unwrap=np.unwrap(dataframe['HeadRotation_X'], period=360), HeadRotation_Y_unwrap=np.unwrap(dataframe['HeadRotation_Y'], period=360), HeadRotation_Z_unwrap=np.unwrap(dataframe['HeadRotation_Z'], period=360))

# Data in korte stukjes knippen
# Vind de tijd x sec verder van vorige laatste index. loopje maken; wanneer eindigt het? als er niet meer de helft van duration_piece beschikbaar is. 
index = []
time = []


for i in range(10):
    duration_piece = 10

    if not index:
        time.append(take_closest(list(df2['Time']), df2['Time'][0]+duration_piece))
# vgm had ik index alleen om de index bij de bijpassende tijd te vinden. is dit wel nodig? het lukt nog niet :(
        index = df2['Time'].get_loc(time[i]) 
        # index = int(((df2.index[df2['Time'] == time[i]])))
        # index.append(df2[df2['Time'] == time[i]].index.values)
        print(index)

    else:
        time.append(take_closest(list(df2['Time']), df2['Time'][index]+duration_piece))
        # time.append(bisect_left(list(df2['Time']), df2['Time'][int(index[-1])]+duration_piece))
        # index.append(df2[df2['Time'] == time[i]].index.values)
        # index.append((df2.index[df2['Time'] == time[i]]))

    # if time[i] < 0.5*duration_piece*time[i-1]:
    #     print('yes')
    #     break

print('end loop')
print(time)

# indexx = [20, 40, 50, 60]
# print(df2['Time'][indexx[-1]])


# Calculate standard deviations of positions x, y, z and rotation x, y and z
# print(np.std(df2['HeadPosition_X']))
# print(np.std(df2['HeadPosition_Y']))
# print(np.std(df2['HeadPosition_Z']))

# print(np.std(df2['HeadRotation_X_unwrap']))
# print(np.std(df2['HeadRotation_Y_unwrap']))
# print(np.std(df2['HeadRotation_Z_unwrap']))

