# Stress analysis

import pandas as pd
import numpy as np
from bisect import bisect_left
from statistics import mean
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


def preprocessing(dataframe):
    # Voor nu: vervangen rotatiematrices en confidance columns verwijderen.

    # Replace rotation columns of dataframe to center values around 360 or 0. Voor nu alleen de rotatie van de 
    # rechterhand toegevoegd.
    # Het lukt nog niet om met een loopje de dataframe te wijzigen omdat hij telkens opnieuw gedefinieerd wordt. 
    # later nog naar kijken
    # for key in rotations:
    #     df2 = dataframe.assign(key=np.unwrap(dataframe[key], period=360))
    df2 = dataframe.assign(HeadRotation_X=np.unwrap(dataframe['HeadRotation_X'], period=360),
                           HeadRotation_Y=np.unwrap(dataframe['HeadRotation_Y'], period=360),
                           HeadRotation_Z=np.unwrap(dataframe['HeadRotation_Z'], period=360),
                           EyeRotationLeft_X=np.unwrap(dataframe['EyeRotationLeft_X'], period=360),
                           EyeRotationLeft_Y=np.unwrap(dataframe['EyeRotationLeft_Y'], period=360),
                           EyeRotationRight_X=np.unwrap(dataframe['EyeRotationRight_X'], period=360),
                           EyeRotationRight_Y=np.unwrap(dataframe['EyeRotationRight_Y'], period=360),
                           HandRotationRight_X=np.unwrap(dataframe['HandRotationRight_X'], period=360),
                           HandRotationRight_Y=np.unwrap(dataframe['HandRotationRight_Y'], period=360),
                           HandRotationRight_Z=np.unwrap(dataframe['HandRotationRight_Z'], period=360))

    # Remove columns that contain 'confidance'
    preprocessed_dataframe = df2[df2.columns.drop(list(df2.filter(regex='Confidance')))]
    # preprocessed_dataframe = dataframe[dataframe.columns.drop(list(dataframe.filter(regex='Confidance')))]
    return preprocessed_dataframe


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    This can be used for finding the start and end time of the data pieces.
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
    for i in range(30):
        if i == 0:
            time = (take_closest(list(dataframe['Time']), (dataframe['Time'][0]+duration_piece)))
            times.append(time)
        else:
            time = (take_closest(list(dataframe['Time']), time+duration_piece))
            # Tijd niet toevoegen als het stukje korter is dan 0.5 x duration piece
            if time - times[-1] < 0.5*duration_piece:
                break
            else:
                times.append(time)
    # Find indices of times
    for j in range(len(times)):
        ind = int(dataframe[dataframe['Time'] == times[j]].index.values)
        indices.append(ind)
    # Create a dict of the dataframes of the different pieces
    for i in range(len(indices)-1):
        d[f"dataframe{person}_{i}"] = dataframe.loc[indices[i]:indices[i+1]-1, :]
    return d


def euclidean_speed(df, parameters):
    # For position features
    distances = []
    time_steps = []
    speed_changes = []
    dataframe = df.reset_index()
    for i in range(len(dataframe['Time'])-1):
        a = np.array([dataframe[parameters[0]][i], dataframe[parameters[1]][i], dataframe[parameters[2]][i]])
        b = np.array([dataframe[parameters[0]][i+1], dataframe[parameters[1]][i+1], dataframe[parameters[2]][i+1]])
        time_step = dataframe['Time'][i+1]-dataframe['Time'][i]
        dist = np.linalg.norm(a-b)
        time_steps.append(time_step)
        distances.append(dist)
    p_speeds = [i / j for i, j in zip(distances, time_steps)]
    # berekenen van de acceleratie op basis van het verschil in snelheid gedeeld door het verschil in tijd (eerder berekende tijdstapjes)
    for j in range(len(p_speeds)-1):
        speed_change = p_speeds[j]-p_speeds[j+1]
        speed_changes.append(speed_change)
    p_accelerations = [i / j for i, j in zip(speed_changes, time_steps)]
    return p_speeds, p_accelerations


def speed(df, parameter):
    # For rotations en face features
    distances = []
    time_steps = []
    dataframe = df.reset_index()
    for i in range(len(dataframe['Time'])-1):
        time_step = dataframe['Time'][i+1]-dataframe['Time'][i]
        # Hier wordt het absolute verschil berekend
        dist = np.linalg.norm(dataframe[parameter][i]-dataframe[parameter][i+1])
        time_steps.append(time_step)
        distances.append(dist)
    rf_speeds = [i / j for i, j in zip(distances, time_steps)]
    return rf_speeds


def scale_data(data_train):
    # scale_data(data_train, data_test)
    keys = data_train.keys()
    # Scale the data to 0-1
    scaler = MinMaxScaler()
    scale_train = scaler.fit_transform(data_train)
    data_train.loc[:, (keys)] = scale_train
    # scale_test = scaler.transform(data_test)
    # data_test.loc[:, (keys)] = scale_train
    return data_train  # , scale_test


# files = ... loopje over alle files in een folder voor het creeeren van verschillende dataframes.
path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/23-03-16 14-56-02 trackingData.csv'
# path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Testset.csv'
df = pd.read_table(path, delimiter=";", dtype=np.float64)

# Remove last rows where time = zero and for now; remove the rows where head position is 0. Dit kan geskipt voor de echte data
dataframe_ = df[df.Time != 0.00000]
dataframe = dataframe_[dataframe_.HeadPosition_X != 0.00000]
df3 = preprocessing(dataframe)

# Dataframes van de verschillende stukjes maken
d = cut_dataframe(df3, 1, 1)

# Keys voor positions
positions = ['HeadPosition_X', 'HeadPosition_Y', 'HeadPosition_Z', 'HandPositionRight_X', 'HandPositionRight_Y',
             'HandPositionRight_Z']

# Keys voor rotations
rotations = ['HeadRotation_X', 'HeadRotation_Y', 'HeadRotation_Z', 'EyeRotationLeft_X', 'EyeRotationLeft_Y',
             'EyeRotationRight_X', 'EyeRotationRight_Y', 'HandRotationRight_X', 'HandRotationRight_Y',
             'HandRotationRight_Z']

# Keys voor gezichtsfeatures
face_features = ['BrowLowererL', 'BrowLowererR', 'CheekPuffL', 'CheekPuffR', 'CheekRaiserL', 'CheekRaiserR',
                 'CheekSuckL', 'CheekSuckR', 'ChinRaiserB', 'ChinRaiserT', 'DimplerL', 'DimplerR', 'EyesClosedL',
                 'EyesClosedR', 'EyesLookDownL', 'EyesLookDownR', 'EyesLookLeftL', 'EyesLookLeftR', 'EyesLookRightL',
                 'EyesLookRightR', 'EyesLookUpL', 'EyesLookUpR', 'InnerBrowRaiserL', 'InnerBrowRaiserR', 'JawDrop',
                 'JawSidewaysLeft', 'JawSidewaysRight', 'JawThrust', 'LidTightenerL', 'LidTightenerR',
                 'LipCornerDepressorL', 'LipCornerDepressorR', 'LipCornerPullerL', 'LipCornerPullerR', 'LipFunnelerLB',
                 'LipFunnelerLT', 'LipFunnelerRB', 'LipFunnelerRT', 'LipPressorL', 'LipPressorR', 'LipPuckerL',
                 'LipPuckerR', 'LipStretcherL', 'LipStretcherR', 'LipSuckLB', 'LipSuckLT', 'LipSuckRB', 'LipSuckRT',
                 'LipTightenerL', 'LipTightenerR', 'LipsToward', 'LowerLipDepressorL', 'LowerLipDepressorR',
                 'MouthLeft', 'MouthRight', 'NoseWrinklerL', 'NoseWrinklerR', 'OuterBrowRaiserL', 'OuterBrowRaiserR',
                 'UpperLidRaiserL', 'UpperLidRaiserR', 'UpperLipRaiserL', 'UpperLipRaiserR']

# Lege dict definiëren
dict_sum = defaultdict(list)

# In deze loop worden voor alle dataframes in de dictionary voor 1 persoon features berekend voor de positions,
# rotations en face features.
for i in list(d.keys()):
    for j in range(len(positions)):
        dict_sum[f"{positions[j]}_std"].append(np.std(d[i][positions[j]]))
    for k in range(len(rotations)):
        dict_sum[f"{rotations[k]}_std"].append(np.std(d[i][rotations[k]]))
        dict_sum[f"{rotations[k]}_speed_mean"].append(mean(speed(d[i], rotations[k])))
        dict_sum[f"{rotations[k]}_speed_std"].append(np.std(speed(d[i], rotations[k])))
    for m in range(len(face_features)):
        dict_sum[f"{face_features[m]}_std"].append(np.std(d[i][face_features[m]]))
        dict_sum[f"{face_features[m]}_speed_mean"].append(mean(speed(d[i], face_features[m])))
        dict_sum[f"{face_features[m]}_speed_std"].append(np.std(speed(d[i], face_features[m])))
    dict_sum["HeadPosition_speed_mean"].append(mean((euclidean_speed(d[i], [positions[0], positions[1], positions[2]])[0])))
    dict_sum["HandPosition_speed_mean"].append(mean((euclidean_speed(d[i], [positions[3], positions[4], positions[5]])[0])))
    dict_sum["HeadPosition_speed_std"].append(np.std((euclidean_speed(d[i], [positions[0], positions[1], positions[2]])[0])))
    dict_sum["HandPosition_speed_std"].append(np.std((euclidean_speed(d[i], [positions[3], positions[4], positions[5]])[0])))
    dict_sum["HeadPosition_acceleration_mean"].append(mean((euclidean_speed(d[i], [positions[0], positions[1], positions[2]])[1])))
    dict_sum["HandPosition_acceleration_mean"].append(mean((euclidean_speed(d[i], [positions[3], positions[4], positions[5]])[1])))
    dict_sum["HeadPosition_acceleration_std"].append(np.std((euclidean_speed(d[i], [positions[0], positions[1], positions[2]])[1])))
    dict_sum["HandPosition_acceleration_std"].append(np.std((euclidean_speed(d[i], [positions[3], positions[4], positions[5]])[1])))

df_sum = pd.DataFrame(data=dict_sum)  # Deze aan het einde, na het berekenen van alle features

# Het combineren van oog features links en rechts en het verwijderen van links en rechts apart
df_sum['EyeRotationLR_X_speed_mean'] = df_sum[['EyeRotationLeft_X_speed_mean', 'EyeRotationRight_X_speed_mean']].mean(axis=1)
df_sum['EyeRotationLR_Y_speed_mean'] = df_sum[['EyeRotationLeft_Y_speed_mean', 'EyeRotationRight_Y_speed_mean']].mean(axis=1)
df_sum['EyeRotationLR_X_speed_std'] = df_sum[['EyeRotationLeft_X_speed_std', 'EyeRotationRight_X_speed_std']].mean(axis=1)
df_sum['EyeRotationLR_Y_speed_std'] = df_sum[['EyeRotationLeft_Y_speed_std', 'EyeRotationRight_Y_speed_std']].mean(axis=1)
df_sum['EyeRotationLR_X_std'] = df_sum[['EyeRotationLeft_X_std', 'EyeRotationRight_X_std']].mean(axis=1)
df_sum['EyeRotationLR_Y_std'] = df_sum[['EyeRotationLeft_Y_std', 'EyeRotationRight_Y_std']].mean(axis=1)
df_sum2 = df_sum.drop(['EyeRotationLeft_X_speed_mean', 'EyeRotationRight_X_speed_mean', 'EyeRotationLeft_Y_speed_mean',
                       'EyeRotationRight_Y_speed_mean', 'EyeRotationLeft_X_speed_std', 'EyeRotationRight_X_speed_std',
                       'EyeRotationLeft_Y_speed_std', 'EyeRotationRight_Y_speed_std', 'EyeRotationLeft_X_std',
                       'EyeRotationRight_X_std', 'EyeRotationLeft_Y_std', 'EyeRotationRight_Y_std'], axis=1)
# df_sum2 moet een naam krijgen als df_{i} ofzo, zodat voor iedere persoon een andere dataframe wordt gecreeerd.

# Hierna moeten alle personen samengevoegd worden in 1 dataframe en worden train en test verdeeld.

scaled_data = scale_data(df_sum2)
