import random

def check_on_leaderboard(ground_truth, submission):
    submission = np.clip(submission, e-15, 1-e-15)
    result = -np.mean(expected*np.log(predicted) + (1-expected)*np.log(1-predicted))
    return float("%.06f"%result)  # round to 6 digits after comma



for i in xrange(1000):
    submission = [random.randint(0,1) for i in xrange()]


