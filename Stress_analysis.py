# Stress analysis

import pandas as pd
import numpy as np
from bisect import bisect_left


def preprocessing(dataframe):
    # Voor nu: vervangen rotatiematrices en confidance columns verwijderen.

    # Replace rotation columns of dataframe to center values around 360 or 0. Voor nu alleen de rotatie van de rechterhand toegevoegd.
    df2 = dataframe.assign(HeadRotation_X=np.unwrap(dataframe['HeadRotation_X'], period=360), HeadRotation_Y=np.unwrap(dataframe['HeadRotation_Y'], period=360), HeadRotation_Z=np.unwrap(dataframe['HeadRotation_Z'], period=360),
                           EyeRotationLeft_X=np.unwrap(dataframe['EyeRotationLeft_X'], period=360), EyeRotationLeft_Y=np.unwrap(dataframe['EyeRotationLeft_Y'], period=360),
                           EyeRotationRight_X=np.unwrap(dataframe['EyeRotationRight_X'], period=360), EyeRotationRight_Y=np.unwrap(dataframe['EyeRotationRight_Y'], period=360),
                           HandRotationRight_X=np.unwrap(dataframe['HandRotationRight_X'], period=360), HandRotationRight_Y=np.unwrap(dataframe['HandRotationRight_Y'], period=360), HandRotationRight_Z=np.unwrap(dataframe['HandRotationRight_Z'], period=360))

    # Remove columns that contain 'confidance'
    preprocessed_dataframe = df2[df2.columns.drop(list(df2.filter(regex='Confidance')))]
    return preprocessed_dataframe


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
        ind = int(dataframe[dataframe['Time'] == times[j]].index.values)
        indices.append(ind)
    # Create a dict of the dataframes of the different pieces
    for i in range(len(indices)-1):
        d[f"dataframe{person}_{i}"] = dataframe.loc[indices[i]:indices[i+1]-1, :]
    return d


def euclidean(dataframe, parameters):
    # For head position and hand position
    distances = []
    for i in range(len(dataframe['Time'])-1):
        a = np.array([dataframe[parameters[0]][i], dataframe[parameters[1]][i], dataframe[parameters[2]][i]])
        b = np.array([dataframe[parameters[0]][i+1], dataframe[parameters[1]][i+1], dataframe[parameters[2]][i+1]])
        dist = np.linalg.norm(a-b)
        distances.append(dist)
        # geeft nog een error
    return distances


path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Testset.csv'
df = pd.read_table(path, delimiter=";", dtype=np.float64)

# Remove last rows where time = zero and for now; remove the rows where head position is 0
dataframe_ = df[df.Time != 0.00000]
dataframe = dataframe_[dataframe_.HeadPosition_X != 0.00000]

# print(dataframe['HeadRotation_X'].head())
df3 = preprocessing(dataframe)

# Dataframes van de verschillende stukjes maken
d = cut_dataframe(df3, 1)

# Euclidean distance berekenen voor posities x, y en z
distances = euclidean(df3, ['HeadPosition_X', 'HeadPosition_Y', 'HeadPosition_Z'])
print(distances)


# # Calculate standard deviations of positions x, y, z and rotation x, y and z
# # print(np.std(df2['HeadPosition_X']))
# # print(np.std(df2['HeadPosition_Y']))
# # print(np.std(df2['HeadPosition_Z']))

# # print(np.std(df2['HeadRotation_X_unwrap']))
# # print(np.std(df2['HeadRotation_Y_unwrap']))
# # print(np.std(df2['HeadRotation_Z_unwrap']))
