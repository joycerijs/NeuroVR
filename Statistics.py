# statistics

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from statistics import mean
from statistics import stdev


def statistics(rest_data, stress_data, sign_features_dfs):
    # Student's t-test for continuous data
    df_p = pd.DataFrame({'Features': rest_data.keys()})
    for key in rest_data.keys():
        _, p = stats.ttest_ind(rest_data[key], stress_data[key])   # Perform the Student's t-test
        df_p.loc[df_p['Features'] == key, 'P-value'] = p    # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_ID = np.round(stress_data[key].mean(), decimals=2)
        std_ID = np.round(stress_data[key].std(), decimals=2)
        mean_no_ID = np.round(rest_data[key].mean(), decimals=2)
        std_no_ID = np.round(rest_data[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std ID'] = f'{mean_ID} ± {std_ID}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std no ID'] = f'{mean_no_ID} ± {std_no_ID}'

    # Find significant p-values by Holm-Bonferroni:
    df_p_sorted = df_p.sort_values(by=['P-value'])    # Sort the values by p-values
    df_p_sorted['Rank'] = range(1, len(df_p_sorted)+1)    # Rank the features
    df_p_sorted['Significance level'] = 0.05/(len(df_p_sorted)+1-df_p_sorted['Rank'])    # Calculate the significance level per feature
    df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < df_p_sorted['Significance level'], 'Yes', 'No')    # Find which features are significant

    # Create dataframe with significant features only and create table for visualisation
    df_p_sign = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']
    df_p_for_table = df_p_sign.drop(['Rank'], axis=1)

    # Append the dataframe with significant features to a list for every fold. In this list, the dataframes for the 10 folds are stored.
    sign_features_dfs.append(df_p_for_table)

    # Create list of significant features that can be used for model creation
    sign = df_p_sign['Features'].tolist()

    return sign, sign_features_dfs


sign_features_dfs = []

file_rest = pd.read_table('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/rust_data_gecombineerd.csv', delimiter=";", decimal=',')
file_stress = pd.read_table('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/stress_data_gecombineerd.csv', delimiter=";", decimal=',')

sign, sign_features_dfs = statistics(file_rest, file_stress, sign_features_dfs)

print(sign_features_dfs)