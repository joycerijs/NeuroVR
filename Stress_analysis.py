# Stress analysis

import pandas as pd
import numpy as np
import os
from bisect import bisect_left
from statistics import mean
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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


def scale_data(data_train, data_test):
    # scale_data(data_train, data_test)
    keys = data_train.keys()
    # Scale the data to 0-1
    scaler = MinMaxScaler()
    scale_train = scaler.fit_transform(data_train)
    data_train.loc[:, (keys)] = scale_train
    scale_test = scaler.transform(data_test)
    data_test.loc[:, (keys)] = scale_test
    return data_train, data_test


def pipeline_model(train_data, train_label, test_data, test_label, clf, tprs, aucs, spec, sens, accuracy, axis):
    '''In this function, a machine learning model is created and tested. Dataframes of the train data, train labels, test data and test labels
    must be given as input. Also, the classifier must be given as input. Scoring metrics true positives, area under curve, specificity, sensitivity
    and accuracy must be given as input, these scores are appended every fold and are returned. The axis must also be given in order to plot the ROC curves
    for the different folds in the right figure.'''
    # Fit and test the classifier
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    # plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)    # Help for plotting the false positive rate
    viz = metrics.plot_roc_curve(clf, test_data, test_label, name='ROC fold {}'.format(i), alpha=0.3, lw=1, ax=axis)    # Plot the ROC-curve for this fold on the specified axis.
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)    # Interpolate the true positive rate
    interp_tpr[0] = 0.0    # Set the first value of the interpolated true positive rate to 0.0
    tprs.append(interp_tpr)   # Append the interpolated true positive rate to the list
    aucs.append(viz.roc_auc)    # Append the area under the curve to the list

    # Calculate the scoring metrics
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()   # Find the true negatives, false positives, false negatives and true positives from the confusion matrix

    spec.append(tn/(tn+fp))    # Append the specificity to the list
    sens.append(tp/(tp+fn))    # Append the sensitivity to the list
    accuracy.append(metrics.accuracy_score(test_label, predicted))    # Append the accuracy to the list

    return tprs, aucs, spec, sens, accuracy, predicted


def mean_ROC_curves(tprs, aucs, axis):
    '''With this function, the mean ROC-curves of the models over a 10-cross-validation are plot.
    The true positive rates, areas under the curve and axes where the mean ROC-curve must be plot
    are given as input for different models. The figures are filled with the mean and std ROC-curve and
    can be visualized with plt.show()'''
    # for i, (tprs, aucs, axis) in enumerate(zip(tprs_all, aucs_all, axis_all)):   # Loop over the tprs, aucs and first three axes for the figures of the three different models.
    # Calculate means and standard deviations of true positive rate, false positive rate and area under curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    axis.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)   # Plot the mean ROC-curve for the corresponding model
    # axis_all.plot(mean_fpr, mean_tpr, label=fr'Mean ROC model {(i+1)} (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)    # Plot the mean ROC-curve for the corresponding model in another figure
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)    # Set the upper value of the true positive rates
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    # Set the upper value of the true positive rates
    axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves
    axis.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='ROC-curves model')    # Set axes and title
    axis.legend(loc="lower right")    # Set legend
    # axis_all.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves in another figure
    # axis_all.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Mean ROC-curve for the three models')    # Set axes and title
    # axis_all.legend(loc="lower right")    # Set legend
    return


path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Alle data'
files = os.listdir(path)
dict_all_files = {}  # Lege dict om straks alle personen in op te slaan
labels = []

for idp, p in enumerate(files):
    # Loop over alle files om dicts te creeren van de features.
    # df = pd.read_table(os.path.join(path, p), delimiter=";", dtype=np.float64)
    df = pd.read_table(os.path.join(path, p), delimiter=";", decimal=',')

    # Remove last rows where time = zero and for now; remove the rows where head position is 0. Dit kan geskipt voor de echte data
    dataframe_ = df[df.Time != 0.00000]
    dataframe = dataframe_.drop(dataframe_[dataframe_.ExpressionConfidanceUpperFace < 0.2].index)  # Missing data rijen verwijderen. die zijn -1. Misschien reset index?
    df3 = preprocessing(dataframe.reset_index())

    # Dataframes van de verschillende stukjes maken
    duration = 30  # Change duration of pieces
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
        dict_sum["HeadPosition_speed_mean"].append(mean((euclidean_speed(d[i], [positions[0], positions[1],
                                                                                positions[2]])[0])))
        dict_sum["HandPosition_speed_mean"].append(mean((euclidean_speed(d[i], [positions[3], positions[4],
                                                                                positions[5]])[0])))
        dict_sum["HeadPosition_speed_std"].append(np.std((euclidean_speed(d[i], [positions[0], positions[1],
                                                                                 positions[2]])[0])))
        dict_sum["HandPosition_speed_std"].append(np.std((euclidean_speed(d[i], [positions[3], positions[4],
                                                                                 positions[5]])[0])))
        dict_sum["HeadPosition_acceleration_mean"].append(mean((euclidean_speed(d[i], [positions[0], positions[1],
                                                                                       positions[2]])[1])))
        dict_sum["HandPosition_acceleration_mean"].append(mean((euclidean_speed(d[i], [positions[3], positions[4],
                                                                                       positions[5]])[1])))
        dict_sum["HeadPosition_acceleration_std"].append(np.std((euclidean_speed(d[i], [positions[0], positions[1],
                                                                                        positions[2]])[1])))
        dict_sum["HandPosition_acceleration_std"].append(np.std((euclidean_speed(d[i], [positions[3], positions[4],
                                                                                        positions[5]])[1])))
        dict_sum['Set'].append(idp)  # voeg een kolom toe met de naam van de set (dus het getal van de file (1 t/m 40 ongeveer))

        if df3['PrevSceneName'][2] == 'Stress':
            dict_sum['Label'].append(1)  # voeg een kolom met het label toe voor iedere window van een set.
        else:
            dict_sum['Label'].append(0)

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
    # dict_all_files[f"{p}"].append(df_sum2)
    # dict_all_files[f"Set {idp}"].append(df_sum2)
    if df3['PrevSceneName'][2] == 'Stress':
        labels.append(1)
    else:
        labels.append(0)

    dict_all_files[f"{idp}"] = df_sum2  # Nog uitzoeken waarom ik dit allebei had

# scaled_data = scale_data(df_sum2)
cv_10fold = model_selection.StratifiedKFold(n_splits=5)

tprs_RF_all = []
aucs_RF_all = []
spec_RF_all = []
sens_RF_all = []
accuracy_RF_all = []
_, axis_RF_all = plt.subplots()

for i, (train_index, test_index) in enumerate(cv_10fold.split(dict_all_files, labels)):
    appended_data_train = []
    appended_data_test = []

    for j in range(len(train_index)):
        data_train = dict_all_files[(list(dict_all_files.keys()))[(train_index[j])]]
        appended_data_train.append(data_train)
    for k in range(len(test_index)):
        data_test = dict_all_files[(list(dict_all_files.keys()))[(test_index[k])]]
        appended_data_test.append(data_test)

    appended_data_train = pd.concat(appended_data_train, ignore_index=True)
    appended_data_test = pd.concat(appended_data_test, ignore_index=True)
    # scaled_train, scaled_test = scale_data(appended_data_train, appended_data_test)
    # train en test staan nu in aparte dataframes, met labels.

    train_label = list(appended_data_train['Label'])
    train_data = appended_data_train.drop(['Label', 'Set'], axis=1)
    test_label = list(appended_data_test['Label'])
    test_data = appended_data_test.drop(['Label', 'Set'], axis=1)

    clf_RF_all = RandomForestClassifier()
    # Random forest with all features: create model
    tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, predicted = pipeline_model(train_data, train_label, test_data, test_label, clf_RF_all, tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, axis_RF_all)

    # Start aan loopje om per set te berekenen hoeveel windows als stress gelabeld moeten worden.
    for m in test_index:
        sum = appended_data_test[appended_data_test['Set'] == m]['Label'].sum()
        dict_predicted = {'Set': list(appended_data_test['Set']), 'Predicted label': predicted}
        df_predicted = pd.DataFrame(data=dict_predicted)
        sum_predicted = df_predicted[df_predicted['Set'] == m]['Predicted label'].sum()
        print(f'sum: {sum}, sum predicted: {sum_predicted}')

# mean_ROC_curves(tprs_RF_all, aucs_RF_all, axis_RF_all)
# plt.show()

dict_scores = {'Model 1: RF with all features': [f'{np.round(mean(accuracy_RF_all), decimals=2)} ± {np.round(np.std(accuracy_RF_all), decimals=2)}',
                                                 f'{np.round(mean(sens_RF_all), decimals=2)} ± {np.round(np.std(sens_RF_all), decimals=2)}',
                                                 f'{np.round(mean(spec_RF_all), decimals=2)} ± {np.round(np.std(spec_RF_all), decimals=2)}',
                                                 f'{np.round(mean(aucs_RF_all), decimals=2)} ± {np.round(np.std(aucs_RF_all), decimals=2)}']}

df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['Accuracy', 'Sensitivity', 'Specificity', 'Area under ROC-curve'])

print(df_scores)
