import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

treshold = 10
student_features = pd.read_csv("student-mat.csv", header=0, delimiter=";")
features = student_features.to_numpy()

# finding all different marginal probabilities in the list
# data_list shape [feature_num]
# returning all different combination of data and their probability


def find_marginal_prob(data_list):

    type_list = []
    data_counter = []

    for data in range(len(data_list)):
        data_type = [data_list[data][value] for value in range(len(data_list[0]))]
        if data_type in type_list:
            data_counter[type_list.index(data_type)] += 1
        else:
            type_list.append(data_type)
            data_counter.append(1)

    data_prob = []
    for counter in data_counter:
        data_prob.append(counter/len(data_list))
    return type_list, data_prob
# find conditional probability at a list by known condition
# data_list's shape is [[feature_num],1] known_info's shape [known_type, known value] desired_info's shape [desired_type, desired value]
# returning just probability


def find_cond_prob(data_list, known_info, desired_info):

    desired_counter = 0
    known_counter = 0
    for value in range(len(data_list)):
        if data_list[value][known_info[0]] == known_info[1]:
            known_counter += 1
            if data_list[value][desired_info[0]] == desired_info[1]:
                desired_counter += 1
    prob = desired_counter/known_counter

    return prob


def train(x_train, y_train):

    data_matched = []
    for data_num in range(len(x_train)):
        data_matched.append([x_train[data_num], y_train[data_num]])

    type_list, prob_list = find_marginal_prob(x_train)
    evidence_probs_list = []
    for prob_num in range(len(type_list)):
        type_prob_pair = [type_list[prob_num], prob_list[prob_num]]
        evidence_probs_list.append(type_prob_pair)
    superior_prob_list = []
    for type_num in type_list:
        cond_prob_pair = [[type_num, 1], find_cond_prob(data_matched, [1, 1], [0, type_num])]
        superior_prob_list.append(cond_prob_pair)

    return evidence_probs_list, superior_prob_list

# superior_probs shape [[[feauture_num], 1], 1], y_test]
# evidence probs shape [[shape_num, 1], 1], y_test]
# returning true positive, true negative, false positive, false negative


def test(superior_probs, evidence_probs, x_test, y_test):
    predictions = []
    for test_data in x_test:
        evidence_prob = 0
        superior_prob = 0
        for type_number in range(len(evidence_probs)):
            if test_data == evidence_probs[type_number][0]:
                evidence_prob = evidence_probs[type_number][1]
                superior_prob = superior_probs[type_number][1]
        prediction = superior_prob*prior_prob/evidence_prob
        if prediction >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for prediction in range(len(y_test)):
        if predictions[prediction] == 1 and y_test[prediction] == 1:
            tp += 1
        elif predictions[prediction] == 1 and y_test[prediction] == 0:
            fn += 1
        elif predictions[prediction] == 0 and y_test[prediction] == 0:
            tn += 1
        else:
            fp += 1
    return tp, tn, fp, fn


feature_indexes = [30, 31, 32]  # ["G1", "G2", "G3"]
feature_for_prob = []
for feature in features:
    feature_for_prob.append([feature[value] for value in feature_indexes])

# transform g1, g2 and g3 to binary form yes or no if value less than 10 then it is no, yes otherwise
final_grades = []
for values in range(len(feature_for_prob)):
    if feature_for_prob[values][-1] <= treshold:
        final_grades.append(1)
    else:
        final_grades.append(0)
    if feature_for_prob[values][0] < 10:
        feature_for_prob[values][0] = "no"
    else:
        feature_for_prob[values][0] = "yes"
    if feature_for_prob[values][1] <= 10:
        feature_for_prob[values][1] = "no"
    else:
        feature_for_prob[values][1] = "yes"
    feature_for_prob[values] = feature_for_prob[values][:-1]
x_train, x_test, y_train, y_test = train_test_split(feature_for_prob, final_grades, test_size=0.2)

passed_count = 0
for grade in y_train:
    if grade == 1:
        passed_count += 1
prior_prob = passed_count/len(y_train)

evidence_probs, superior_probs = train(x_train, y_train)
tp, tn, fp, fn = test(superior_probs, evidence_probs, x_test, y_test)
print(str("\n   ")+str("p") + "  |" + str("  ") + str("n"))
print(str("t") + "| " + str(tp) + str("   ") + str(fn))
print(str("f") + "| " + str(fp) + str("   ") + str(tn))

jaccard_coefficient = tp/(fp+tp+fn)

print("\nJaccard Coefficient: ", jaccard_coefficient)
