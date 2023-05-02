# Statistiek

import pandas as pd
import numpy as np
import os
from bisect import bisect_left
from statistics import mean
from collections import defaultdict
from scipy.signal import find_peaks
from scipy import stats


def preprocessing(dataframe):
    '''In deze functie worden de rotatieparameters genormaliseerd rond 360 graden.
    Ook worden de parameters verwijderd waar 'Confidance' wordt genoemd, deze worden niet gebruikt bij de
    data-verwerking. De input van deze functie is een dataframe, de output is de voorbewerkte dataframe.'''
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
    preprocessed_dataframe = df2[df2.columns.drop(list(df2.filter(regex='Confidance')))]
    return preprocessed_dataframe


def take_closest(myList, myNumber):
    '''Deze functie kan gebruikt worden om het dichtst bij zijnde getal van een gegeven getal (myNumber)
    in een list te vinden (myList). Ik gebruik deze functie voor het opdelen van de dataframe in kleinere stukken.
    De output van deze functie is het dichtstbijzijnde getal in de lijst.'''
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


def cut_dataframe(dataframe, person, duration_piece=180):
    '''In deze functie kan een dataframe in kleinere stukken worden geknipt. De inputs zijn een dataframe, de code van
    een persoon en de gewenste tijdsduur van de stukjes. De output is een dictionary voor een persoon met de
    verschillende stukjes erin als dataframe.'''
    times = []
    indices = []
    d = {}
    times.append(dataframe['Time'][0])
    # Vind de start- en eindtijd van ieder stukje
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
    for j in range(len(times)):
        ind = int(dataframe[dataframe['Time'] == times[j]].index.values)
        indices.append(ind)
    for i in range(len(indices)-1):
        d[f"dataframe{person}_{i}"] = dataframe.loc[indices[i]:indices[i+1]-1, :]
    return d


def euclidean_speed_acc(df, parameters):
    '''In deze functie kunnen de snelheid en versnelling van positieparameters kunnen berekend op basis van
    Euclidische ruimte. De inputs zijn een dataframe en de parameter waarvoor de snelheid en versnelling berekend
    moet worden. De outputs bestaan uit lists van de snelheden en versnellingen.'''
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
    for j in range(len(p_speeds)-1):
        speed_change = p_speeds[j]-p_speeds[j+1]
        speed_changes.append(speed_change)
    p_accelerations = [i / j for i, j in zip(speed_changes, time_steps)]
    return p_speeds, p_accelerations


def speed(df, parameter):
    '''In deze functie kan de snelheid van parameters kunnen berekend per dimensie, dus niet Euclidisch.
    De input zijn een dataframe en de parameter waarvoor de snelheid berekend moet worden. De output
    bestaat uit een list met snelheden.'''
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


def feature_dict(path):
    '''In deze functie worden alle variabelen berekend voor iedere file in het gegeven pad. De output van deze functie
    is een dictionary met daarin de berekende variabelen van iedere file.'''
    files = os.listdir(path)
    dict_all_files = {}  # Lege dict om straks alle personen in op te slaan
    for idp, p in enumerate(files):
        # Loop over alle files om dicts te creeren van de features.
        df = pd.read_table(os.path.join(path, p), delimiter=";", decimal=',')
        dataframe = df.drop(df[df.ExpressionConfidanceUpperFace < 0.2].index)  # Missing data rijen verwijderen.
        df3 = preprocessing(dataframe.reset_index())
        # Dataframes van de verschillende stukjes maken. Zo is het mogelijk om over de verschillende stukjes
        # statistiek te berekenen. Ik heb alleen over drie minuten data statistiek berekend.
        duration = 180
        d = cut_dataframe(df3, idp, duration)
        # Keys voor positions
        positions = ['HeadPosition_X', 'HeadPosition_Y', 'HeadPosition_Z', 'HandPositionRight_X', 'HandPositionRight_Y',
                     'HandPositionRight_Z']
        # Keys voor rotations
        rotations = ['HeadRotation_X', 'HeadRotation_Y', 'HeadRotation_Z', 'EyeRotationLeft_X', 'EyeRotationLeft_Y',
                     'EyeRotationRight_X', 'EyeRotationRight_Y', 'HandRotationRight_X', 'HandRotationRight_Y',
                     'HandRotationRight_Z']
        # Keys voor gezichtsfeatures
        face_features = ['BrowLowererL', 'BrowLowererR', 'CheekPuffL', 'CheekPuffR', 'CheekRaiserL', 'CheekRaiserR',
                         'CheekSuckL', 'CheekSuckR', 'ChinRaiserB', 'ChinRaiserT', 'DimplerL', 'DimplerR',
                         'EyesClosedL', 'EyesClosedR', 'EyesLookDownL', 'EyesLookDownR', 'EyesLookLeftL',
                         'EyesLookLeftR', 'EyesLookRightL', 'EyesLookRightR', 'EyesLookUpL', 'EyesLookUpR',
                         'InnerBrowRaiserL', 'InnerBrowRaiserR', 'JawDrop', 'JawSidewaysLeft', 'JawSidewaysRight',
                         'JawThrust', 'LidTightenerL', 'LidTightenerR', 'LipCornerDepressorL', 'LipCornerDepressorR',
                         'LipCornerPullerL', 'LipCornerPullerR', 'LipFunnelerLB', 'LipFunnelerLT', 'LipFunnelerRB',
                         'LipFunnelerRT', 'LipPressorL', 'LipPressorR', 'LipPuckerL', 'LipPuckerR', 'LipStretcherL',
                         'LipStretcherR', 'LipSuckLB', 'LipSuckLT', 'LipSuckRB', 'LipSuckRT', 'LipTightenerL',
                         'LipTightenerR', 'LipsToward', 'LowerLipDepressorL', 'LowerLipDepressorR', 'MouthLeft',
                         'MouthRight', 'NoseWrinklerL', 'NoseWrinklerR', 'OuterBrowRaiserL', 'OuterBrowRaiserR',
                         'UpperLidRaiserL', 'UpperLidRaiserR', 'UpperLipRaiserL', 'UpperLipRaiserR']
        symmetry_features = ['BrowLowerer', 'CheekPuff', 'CheekRaiser', 'CheekSuck', 'Dimpler', 'InnerBrowRaiser',
                             'LidTightener', 'LipCornerDepressor', 'LipCornerPuller', 'LipPressor', 'LipPucker',
                             'LipStretcher', 'LipTightener', 'LowerLipDepressor', 'NoseWrinkler', 'OuterBrowRaiser',
                             'UpperLidRaiser', 'UpperLipRaiser']
        dict_sum = defaultdict(list)
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
                # Calculate frequency of face features
                s = d[i][f"{face_features[m]}"]
                peaks_indices = find_peaks(s)[0]
                peaks = np.array(list(zip(peaks_indices, s[peaks_indices])))
                if list(s[peaks_indices]):
                    threshold_peak = 0.7 * max(s[peaks_indices])
                    filtered_peaks_indices = [index for index, value in peaks if value > threshold_peak]
                    dict_sum[f"{face_features[m]}_f"].append(len(filtered_peaks_indices))
                else:
                    dict_sum[f"{face_features[m]}_f"].append(0)
            dict_sum["HeadPosition_speed_mean"].append(mean((euclidean_speed_acc(d[i], [positions[0], positions[1],
                                                                                        positions[2]])[0])))
            dict_sum["HandPosition_speed_mean"].append(mean((euclidean_speed_acc(d[i], [positions[3], positions[4],
                                                                                        positions[5]])[0])))
            dict_sum["HeadPosition_speed_std"].append(np.std((euclidean_speed_acc(d[i], [positions[0], positions[1],
                                                                                         positions[2]])[0])))
            dict_sum["HandPosition_speed_std"].append(np.std((euclidean_speed_acc(d[i], [positions[3], positions[4],
                                                                                         positions[5]])[0])))
            dict_sum["HeadPosition_acceleration_mean"].append(mean((euclidean_speed_acc(d[i], [positions[0],
                                                                                               positions[1],
                                                                                               positions[2]])[1])))
            dict_sum["HandPosition_acceleration_mean"].append(mean((euclidean_speed_acc(d[i], [positions[3],
                                                                                               positions[4],
                                                                                               positions[5]])[1])))
            dict_sum["HeadPosition_acceleration_std"].append(np.std((euclidean_speed_acc(d[i], [positions[0],
                                                                                                positions[1],
                                                                                                positions[2]])[1])))
            dict_sum["HandPosition_acceleration_std"].append(np.std((euclidean_speed_acc(d[i], [positions[3],
                                                                                                positions[4],
                                                                                                positions[5]])[1])))
            # Voeg features toe van het aantal goede en foute antwoorden
            list_correct = list(d[i]['CorrectAnswers'])
            list_wrong = list(d[i]['WrongAnswers'])
            sum_correct = []
            sum_wrong = []
            for n in range(len(list_correct)-1):
                if list_correct[n] < list_correct[n+1]:
                    sum_correct.append(1)
            for o in range(len(list_wrong)-1):
                if list_wrong[o] < list_wrong[o+1]:
                    sum_wrong.append(1)
            dict_sum['Wrong_answers'].append(len(sum_wrong))
            dict_sum['Correct_answers'].append(len(sum_correct))
            # Voeg symmetry features toe
            for p in range(len(symmetry_features)):
                dict_sum[f"{symmetry_features[p]}_stdsym"].append(np.std(list((d[i][f"{symmetry_features[p]}L"] -
                                                                               d[i][f"{symmetry_features[p]}R"]))))
        df_sum = pd.DataFrame(data=dict_sum)
        # Het combineren van oog features links en rechts en het verwijderen van links en rechts apart
        df_sum['EyeRotationLR_X_speed_mean'] = df_sum[['EyeRotationLeft_X_speed_mean',
                                                       'EyeRotationRight_X_speed_mean']].mean(axis=1)
        df_sum['EyeRotationLR_Y_speed_mean'] = df_sum[['EyeRotationLeft_Y_speed_mean',
                                                       'EyeRotationRight_Y_speed_mean']].mean(axis=1)
        df_sum['EyeRotationLR_X_speed_std'] = df_sum[['EyeRotationLeft_X_speed_std',
                                                      'EyeRotationRight_X_speed_std']].mean(axis=1)
        df_sum['EyeRotationLR_Y_speed_std'] = df_sum[['EyeRotationLeft_Y_speed_std',
                                                      'EyeRotationRight_Y_speed_std']].mean(axis=1)
        df_sum['EyeRotationLR_X_std'] = df_sum[['EyeRotationLeft_X_std', 'EyeRotationRight_X_std']].mean(axis=1)
        df_sum['EyeRotationLR_Y_std'] = df_sum[['EyeRotationLeft_Y_std', 'EyeRotationRight_Y_std']].mean(axis=1)
        df_sum2 = df_sum.drop(['EyeRotationLeft_X_speed_mean', 'EyeRotationRight_X_speed_mean',
                               'EyeRotationLeft_Y_speed_mean', 'EyeRotationRight_Y_speed_mean',
                               'EyeRotationLeft_X_speed_std', 'EyeRotationRight_X_speed_std',
                               'EyeRotationLeft_Y_speed_std', 'EyeRotationRight_Y_speed_std', 'EyeRotationLeft_X_std',
                               'EyeRotationRight_X_std', 'EyeRotationLeft_Y_std', 'EyeRotationRight_Y_std'], axis=1)
        dict_all_files[f"{idp}"] = df_sum2
    dict = pd.concat(dict_all_files, ignore_index=True)
    return dict


def statistics(rest_data, stress_data):
    '''In deze functie worden alle variabelen in de gegeven files met elkaar vergeleken middels Student's t-test.
    De variabelen die significant van elkaar verschillen (p<0.05) worden in een dataframe gezet met gemiddelde,
    standaard deviatie en p-waarde. Deze dataframe is de output van de functie'''
    df_p = pd.DataFrame({'Features': rest_data.keys()})
    for key in rest_data.keys():
        _, p = stats.ttest_ind(rest_data[key], stress_data[key])
        df_p.loc[df_p['Features'] == key, 'P-value'] = p
        mean_stress = np.round(stress_data[key].mean(), decimals=2)
        std_stress = np.round(stress_data[key].std(), decimals=2)
        mean_rest = np.round(rest_data[key].mean(), decimals=2)
        std_rest = np.round(rest_data[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std stress'] = f'{mean_stress} ± {std_stress}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std rest'] = f'{mean_rest} ± {std_rest}'
    df_p_sorted = df_p.sort_values(by=['P-value'])
    df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < 0.05, 'Yes', 'No')
    df_p_for_table = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']
    return df_p_for_table


path_rest = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Rust data zonder 0'
path_stress = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Stress data zonder 0'
feature_dict_rest = feature_dict(path_rest)
feature_dict_stress = feature_dict(path_stress)
print(feature_dict_rest)
sign_features_df = statistics(feature_dict_rest, feature_dict_stress)
print(sign_features_df)
