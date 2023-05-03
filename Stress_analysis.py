'''Dit Python-bestand kan gebruikt worden om variabelen te berekenen en met deze variabelen een machine learning
model voor stress-detectie te trainen en te testen. Ook kunnen in dit bestand leercurves worden gemaakt, en de
verschillende sub-analyses beschreven in het eindrapport (sectie 3.2.4) kunnen worden uitgevoerd.'''

import pandas as pd
import numpy as np
import os
from bisect import bisect_left
from statistics import mean
from collections import defaultdict
from sklearn import model_selection
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve


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
    '''In deze functie kan een dataframe in kleinere stukken worden geknipt voor de sub-analyses. De inputs zijn
    een dataframe, de code van een persoon en de gewenste tijdsduur van de stukjes. Standaard staat de tijdsduur
    op drie minuten, de data wordt dan niet in stukken geknipt. De output is een dictionary voor een persoon met
    de verschillende stukjes erin als dataframe. Het is ook mogelijk om met deze functie de sub-analyse uit te voeren
    voor het samenvatten van enkel de eerste x-aantal seconden aan data. Het aangegeven stuk moet dan gecomment
    worden.'''
    times = []
    indices = []
    d = {}
    times.append(dataframe['Time'][0])
    # Vind de start- en eindtijd van ieder stukje
    for i in range(30):
        if i == 0:
            time = (take_closest(list(dataframe['Time']), (dataframe['Time'][0]+duration_piece)))
            times.append(time)
        # Om alleen de eerste x-aantal seconden aan data mee te nemen, comment dan het
        # volgende stuk t/m else: times.append(time).
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


def feature_dict(path, duration=180):
    '''In deze functie worden alle variabelen berekend voor iedere file in het gegeven pad. Alleen de lichaams- en
    oogparameters worden berekend. De outputs van deze functie zijn een dictionary met daarin de berekende variabelen
    van iedere file en de bijbehorende labels.'''
    files = os.listdir(path)
    dict_all_files = {}  # Lege dict om straks alle personen in op te slaan
    labels = []
    for idp, p in enumerate(files):
        # Loop over alle files om dicts te creeren van de features.
        df = pd.read_table(os.path.join(path, p), delimiter=";", decimal=',')
        dataframe = df.drop(df[df.ExpressionConfidanceUpperFace < 0.2].index)  # Missing data rijen verwijderen.
        df3 = preprocessing(dataframe.reset_index())
        # Dataframes van de verschillende stukjes maken
        d = cut_dataframe(df3, idp, duration)
        positions = ['HeadPosition_X', 'HeadPosition_Y', 'HeadPosition_Z', 'HandPositionRight_X', 'HandPositionRight_Y',
                     'HandPositionRight_Z']
        rotations = ['HeadRotation_X', 'HeadRotation_Y', 'HeadRotation_Z', 'EyeRotationLeft_X', 'EyeRotationLeft_Y',
                     'EyeRotationRight_X', 'EyeRotationRight_Y', 'HandRotationRight_X', 'HandRotationRight_Y',
                     'HandRotationRight_Z']
        dict_sum = defaultdict(list)
        for i in list(d.keys()):
            for j in range(len(positions)):
                dict_sum[f"{positions[j]}_std"].append(np.std(d[i][positions[j]]))
            for k in range(len(rotations)):
                dict_sum[f"{rotations[k]}_std"].append(np.std(d[i][rotations[k]]))
                dict_sum[f"{rotations[k]}_speed_mean"].append(mean(speed(d[i], rotations[k])))
                dict_sum[f"{rotations[k]}_speed_std"].append(np.std(speed(d[i], rotations[k])))
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
            dict_sum['Set'].append(idp)  # voeg een kolom toe met de naam van de set (dus het getal van de file)
            if df3['PrevSceneName'][2] == 'Stress':
                dict_sum['Label'].append(1)  # voeg een kolom met het label toe voor iedere window van een set.
            else:
                dict_sum['Label'].append(0)
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
        if df3['PrevSceneName'][2] == 'Stress':
            labels.append(1)
        else:
            labels.append(0)
        dict_all_files[f"{idp}"] = df_sum2
    return dict_all_files, labels


def pipeline_model(train_data, train_label, test_data, test_label, clf, tns, tps, fps, fns, spec, sens, accuracy):
    '''In deze functie wordt een machine learning model ontwikkeld en getest. Dataframes met de train data, train
    labels, test data en test labels moeten als input worden gegeven. Metrics terecht-positieven (tp),
    terecht-negatieven (tn), fout-positieven (fp), fout-negatieven (fn), sensitiviteit, specificiteit en
    accuraatheid worden als input gegeven, aangevuld bij elke fold van de cross-validatie en als output gegeven.'''
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    tns.append(tn)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)
    spec.append(tn/(tn+fp))
    sens.append(tp/(tp+fn))
    accuracy.append(metrics.accuracy_score(test_label, predicted))
    return tns, tps, fps, fns, spec, sens, accuracy


def plot_learning_curve(estimator, title, X, y, axes, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """ Met deze functie kunnen leercurves worden gemaakt. Deze functie is grotendeels overgenomen van
    https://scikit-learn.org/0.23/auto_examples/model_selection/plot_learning_curve.html.
    Dit bijbehorende parameterbeschrijving is overgenomen van deze website:

    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    ylim = (0.3, 1.01)
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Voorbeelden trainset")
    axes.set_ylabel("Score")
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean,
                      train_scores_mean, alpha=0.1,
                      color="r")
    axes.fill_between(train_sizes, test_scores_mean,
                      test_scores_mean, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
              label="Train score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
              label="Cross-validatie score")
    axes.legend(loc="best")
    return plt


path = 'F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/Alle data'
files = os.listdir(path)
durations = [180]  # Vul hier de verschillende tijdstukken in om te testen voor de sub-analyse.

for duration in durations:
    dict_all_files, labels = feature_dict(path, duration)
    cv = model_selection.StratifiedKFold(n_splits=17)
    tns_RF = []
    tps_RF = []
    fns_RF = []
    fps_RF = []
    spec_RF = []
    sens_RF = []
    accuracy_RF = []

    for i, (train_index, test_index) in enumerate(cv.split(dict_all_files, labels)):
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
        train_label = list(appended_data_train['Label'])
        train_data = appended_data_train.drop(['Label', 'Set'], axis=1)
        test_label = list(appended_data_test['Label'])
        test_data = appended_data_test.drop(['Label', 'Set'], axis=1)

        # Creeer Random Forest model
        clf_RF = RandomForestClassifier()
        tns_RF, tps_RF, fps_RF, fns_RF, spec_RF, sens_RF, accuracy_RF = \
            pipeline_model(train_data, train_label, test_data, test_label, clf_RF, tns_RF, tps_RF, fps_RF, fns_RF,
                           spec_RF, sens_RF, accuracy_RF)

    dict_scores = {'Model scores RF': [np.round(mean(accuracy_RF), decimals=2),
                                       np.round(mean(sens_RF), decimals=2),
                                       np.round(mean(spec_RF), decimals=2),
                                       sum(tns_RF), sum(tps_RF), sum(fps_RF), sum(fns_RF)]}

    df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['Mean accuracy', 'Mean sensitivity',
                                                                             'Mean specificity', 'True negatives',
                                                                             'True positives', 'False positives',
                                                                             'False negatives'])
    print(df_scores)

    # Learning curves: uncomment om deze te plotten
    # estimators = [50, 100, 150, 200]
    # num = 0
    # fig = plt.figure()

    # for estimator in estimators:
    #     ax = fig.add_subplot(2, 2, num + 1)
    #     plot_learning_curve(RandomForestClassifier(n_estimators=estimator),
    #                         f'Leercurve Random Forest n = {estimator}', train_data, train_label, ax, cv)
    #     num += 1
    #     print(num)
    # plt.show()
