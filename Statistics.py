# statistics

import os
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from statistics import mean
from statistics import stdev


def statistics(rest_data, stress_data):
    # Student's t-test for continuous data
    df_p = pd.DataFrame({'Features': rest_data.keys()})
    for key in rest_data.keys():
        _, p = stats.ttest_ind(rest_data[key], stress_data[key])   # Perform the Student's t-test
        df_p.loc[df_p['Features'] == key, 'P-value'] = p    # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_stress = np.round(stress_data[key].mean(), decimals=2)
        std_stress = np.round(stress_data[key].std(), decimals=2)
        mean_rest = np.round(rest_data[key].mean(), decimals=2)
        std_rest = np.round(rest_data[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std stress'] = f'{mean_stress} ± {std_stress}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std rest'] = f'{mean_rest} ± {std_rest}'

    # Find significant p-values and create table with significant features only:
    df_p_sorted = df_p.sort_values(by=['P-value'])    # Sort the values by p-values
    df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < 0.05, 'Yes', 'No')    # Find which features are significant
    df_p_for_table = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']

    return df_p_for_table


file_rest = pd.read_table('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/rust_data_gecombineerd.csv', delimiter=",")
file_stress = pd.read_table('F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 3/Neuro VR/stress_data_gecombineerd.csv', delimiter=",")

sign_features_dfs = statistics(file_rest, file_stress)

print(sign_features_dfs)



# df = pd.read_table('/media/testdata.csv', delimiter=";", decimal=',')

# list_correct = list(df['CorrectAnswers'])
# list_wrong = list(df['WrongAnswers'])
# sum_correct = []
# sum_wrong = []

# # answers = [1 for i in list(df['CorrectAnswers']) if i < ]

# for i in range(len(list_correct)-1):
#     if list_correct[i] < list_correct[i+1]:
#       sum_correct.append(1)

# for i in range(len(list_wrong)-1):
#     if list_wrong[i] < list_wrong[i+1]:
#       sum_wrong.append(1)