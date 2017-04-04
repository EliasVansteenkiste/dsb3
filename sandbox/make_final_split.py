import numpy as np
import hashlib

import utils
import utils_lung
import pathfinder

rng = np.random.RandomState(42)


tvt_ids = utils.load_pkl(pathfinder.VALIDATION_SPLIT_PATH)
train_pids, valid_pids, test_pids = tvt_ids['training'], tvt_ids['validation'], tvt_ids['test']
all_pids = train_pids + valid_pids + test_pids
print 'total number of pids', len(all_pids)

id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
id2label_test = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
id2label.update(id2label_test)
n_patients = len(id2label)

pos_ids = []
neg_ids = []

for pid, label in id2label.iteritems():
	if label:
		pos_ids.append(pid)
	else:
		neg_ids.append(pid)

pos_ratio = 1. * len(pos_ids) / n_patients
print 'pos id ratio', pos_ratio

split_ratio = 0.15
n_target_split = int(np.round(split_ratio*n_patients))
print 'given split ratio', split_ratio
print 'target split ratio', 1. * n_target_split / n_patients

n_pos_ftest = int(np.round(split_ratio*len(pos_ids)))
n_neg_ftest = int(np.round(split_ratio*len(neg_ids)))

final_pos_test = rng.choice(pos_ids,n_pos_ftest, replace=False)
final_neg_test = rng.choice(neg_ids,n_neg_ftest, replace=False)
final_test = np.append(final_pos_test,final_neg_test)
print 'pos id ratio final test set', 1.*len(final_pos_test) / (len(final_test))

final_train = []
final_pos_train = []
final_neg_train = []
for pid in all_pids:
	if pid not in final_test:
		final_train.append(pid)
		if id2label[pid]:
			final_pos_train.append(pid)
		else:
			final_neg_train.append(pid)


print 'pos id ratio final train set', 1.*len(final_pos_train) / (len(final_train))
print 'final test/(train+test):', 1.*len(final_test) / (len(final_train) + len(final_test))

concat_str = ''.join(final_test)
print 'md5 of concatenated pids:', hashlib.md5(concat_str).hexdigest()

output = {'train':final_train, 'test':final_test}
output_name = pathfinder.METADATA_PATH+'final_split.pkl'
utils.save_pkl(output, output_name)
print 'final split saved at ', output_name




