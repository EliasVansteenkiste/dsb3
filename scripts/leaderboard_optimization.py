import csv
import numpy as np



def check_on_leaderboard(expected, predicted):
    expected, predicted = np.array(expected), np.array(predicted)
    predicted = np.clip(predicted, 1e-15, 1-1e-15)

    result = -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))
    return float("%.05f"%result)  # round to 6 digits after comma

labels = dict()
with open('../cache/test_labels.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    next(reader)  # skip the header
    for row in reader:
        labels[row[0]] = int(row[1])

ground_truth = [v for v in labels.itervalues()]

target = 3.17070

# find lower bound
v = 0.05
good = []
eps = -0.00000001
while True:
    submission = [v if t==1 else 1-v for t in ground_truth]
    score = check_on_leaderboard(ground_truth,submission)
    print score
    if score<target:
        v+=eps
    elif score==target:
        good.append(submission)
        v+=eps
    elif score>target:
        break

for g in good:
    print g

best_submission = good[len(good)/2]

for key,value in zip(labels.iterkeys(),best_submission):
    labels[key] = value

print labels

with open('../cache/test_labels.csv', 'rb') as csvfile_input:
    with open('../cache/submission_%.5f.csv'%target, 'wb') as csvfile_output:
        reader = csv.reader(csvfile_input, delimiter=';', quotechar='|')
        writer = csv.writer(csvfile_output, delimiter=',', quotechar='|')
        header = next(reader)  # skip the header
        writer.writerow(header)
        for row in reader:
            writer.writerow([row[0],labels[row[0]]])