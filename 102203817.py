# -*- coding: utf-8 -*-
"""102203817.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ZP8A82Dmgfcrx6DSbF6VC4OG-RDKByRV
"""

import pandas as pd
from google.colab import files
import numpy as np

uploaded = files.upload()

df = pd.read_excel('102203817-data.xlsx')

def normalize(matrix):
    norm_matrix = matrix / np.sqrt((matrix**2).sum())
    return norm_matrix

def weights(matrix, weights):
    return matrix * weights

def ideal_sol(matrix, is_benefit):
    ideal = matrix.max() if is_benefit else matrix.min()
    negative_ideal = matrix.min() if is_benefit else matrix.max()
    return ideal, negative_ideal

def calculate_topsis_scores(matrix, weights, is_benefit_criteria):
    normalized_matrix = normalize(matrix)
    weighted_matrix = weights(normalized_matrix, weights)
    ideal_solutions = [ideal_sol(weighted_matrix.iloc[:, i], is_benefit_criteria[i])
                       for i in range(weighted_matrix.shape[1])]
    dist_to_ideal = np.sqrt(((weighted_matrix - [ideal[0] for ideal in ideal_solutions])**2).sum(axis=1))
    dist_to_negative_ideal = np.sqrt(((weighted_matrix - [ideal[1] for ideal in ideal_solutions])**2).sum(axis=1))

    topsis_scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)
    return topsis_scores

weights = np.array([0.4, 0.3, 0.2, 0.1])
is_benefit_criteria = [True, True, False, True]
criteria_columns = df.columns[1:]
matrix = df[criteria_columns]

df['TOPSIS Score'] = calculate_topsis_scores(matrix, weights, is_benefit_criteria)
df['Rank'] = df['TOPSIS Score'].rank(ascending=False)

df.to_csv("102203817-result", index=False)
files.download("102203817-result.csv")