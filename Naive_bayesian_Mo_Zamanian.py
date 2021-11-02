# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %% Naive Bayes Classification Code From The Scratch

# %% part 0 - import necessary modules

import numpy as np
import pandas as pd  # pandas is only used for loading excel files

# %% part 1 - load and pre-process datasets

#  part 1.1 - load datasets from file

training_dataset = pd.read_excel('training_data.xlsx').values
testing_dataset = pd.read_excel('testing_data.xlsx').values

#  part 1.2 - process raw data

y_tr_raw = training_dataset[::, 2]
y_tr_logical = y_tr_raw == 'private '
X_tr_raw = np.delete(training_dataset, [0, 2], 1)

X_tr_1 = X_tr_raw[y_tr_logical, :]
X_tr_0 = X_tr_raw[np.logical_not(y_tr_logical), :]

y_te_raw = testing_dataset[::, 2]
y_te_logical = y_te_raw == 'private '
X_te_raw = np.delete(testing_dataset, [0, 2], 1)

# %% part 2 - calculate basic class (prior) probabilities

num_classes = 2
class_prior_probs = []
for class_idx in range(num_classes):
    class_prior_probs.append(sum(y_tr_logical) / len(y_tr_logical))
print()
print('class_prior_probs:', class_prior_probs)
print('----------------------------------------------------------------------')

# %% part 3 - calculate conditional probabilities

class_cond_probs = {0: {}, 1: {}}

X_tr_0_mean = np.mean(X_tr_0[:, 1:], axis=0)
X_tr_0_var = np.var(X_tr_0[:, 1:], axis=0)
class_cond_probs[0]['mean vector'] = X_tr_0_mean
class_cond_probs[0]['variance vector'] = X_tr_0_var

X_tr_1_mean = np.mean(X_tr_1[:, 1:], axis=0)
X_tr_1_var = np.var(X_tr_1[:, 1:], axis=0)
class_cond_probs[1]['mean vector'] = X_tr_1_mean
class_cond_probs[1]['variance vector'] = X_tr_1_var

class_cond_probs[0] = {'u-s': {}}; class_cond_probs[1] = {'u-s': {}}
state_list = set(X_tr_raw[:, 0]); counter = -1
for a_X_tr in [X_tr_0[:, 0], X_tr_1[:, 0]]:
    counter += 1
    for state_name in state_list:
        prob = sum(a_X_tr == state_name) / len(a_X_tr)
        class_cond_probs[counter]['u-s'][state_name] = prob

print('class_cond_probs:\n', class_cond_probs)
print('----------------------------------------------------------------------')

# %% part 4 - predict outputs for test samples

prediction_probs = {0: [], 1: []}

#  part 4.1 - start the loop over test samples

for test_idx in range(X_te_raw.shape[0]):
    test_sample = X_te_raw[test_idx]
    test_state = test_sample[0]
    test_quant_features = test_sample[1:]

#  part 4.2 - calculate scores/probabilities for class 0

    class_0_total_cond_prob = class_prior_probs[0]

    for a_state_name, a_state_prob in \
            class_cond_probs[0]['u-s'].items():
        if test_state == a_state_name:
            class_0_test_state_prob = a_state_prob

    class_0_total_cond_prob *= class_0_test_state_prob

    class_0_quant_prod_probs = 1
    for feature, mu, var in \
            zip(test_quant_features, X_tr_0_mean, X_tr_0_var):
        single_prob = 1 / np.sqrt(var) * np.exp(-(feature-mu)**2 / (2*var))
        class_0_quant_prod_probs *= single_prob

    class_0_total_cond_prob *= class_0_quant_prod_probs

    prediction_probs[0].append(class_0_total_cond_prob)

#  part 4.3 - calculate scores/probabilities for class 1

    class_1_total_cond_prob = class_prior_probs[1]

    for a_state_name, a_state_prob in \
            class_cond_probs[1]['u-s'].items():
        if test_state == a_state_name:
            class_1_test_state_prob = a_state_prob

    class_1_total_cond_prob *= class_1_test_state_prob

    class_1_quant_prod_probs = 1
    for feature, mu, var in zip(test_quant_features, X_tr_1_mean, X_tr_1_var):
        single_prob = 1 / np.sqrt(var) * np.exp(-(feature - mu) ** 2 / (2 * var))
        class_1_quant_prod_probs *= single_prob

    class_1_total_cond_prob *= class_1_quant_prod_probs

    prediction_probs[1].append(class_1_total_cond_prob)

# %% part 5 - evaluate prediction results

print("(raw) prediction_probs:\n", prediction_probs)

y_prediction_logical = np.greater(prediction_probs[1],
                                     prediction_probs[0])

y_te_logical = y_te_logical + 0
y_prediction_logical = y_prediction_logical + 0

print('----------------------------------------------------------------------')
print('performance on samples:')
print('-----------------------')
print('original classes:\n', (y_te_logical))
print('proposed classes:\n', (y_prediction_logical))
print('----------------------------------------------------------------------')
print('----------------------------------------------------------------------')

# %% part 6 - generate confusion matrix

tp = np.dot(y_te_logical, y_prediction_logical)
tn = np.dot(1-y_te_logical, 1-y_prediction_logical)

p = sum(y_te_logical)  # p = tp + fn
n = len(y_te_logical) - p  # n = tn + fp

fp = n - tn; fn = p - tp

print('Confusion Matrix:')
print('----------------------------------------------------------------------')
print(f'Pos (private)= {p}, Neg (public)= {n} :')
print('----------------------------------------')
print(f'TP = {tp}, FP = {fp};\nFN = {fn}, TN = {tn};')


