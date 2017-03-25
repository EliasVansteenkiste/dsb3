import pathfinder
import numpy as np
import utils
import utils_lung
import csv



def write_aapm_candidates(candidates_path):

    

    aapm_id2candidates_path = utils_lung.get_candidates_paths(candidates_path)

    #ok, now we have all the candidate paths
    # figure out how the labels are loaded

    aapm_id2label=utils_lung.read_aapm_annotations(pathfinder.AAPM_LABELS_PATH)
    patient_ids=aapm_id2label.keys()

    pid2negcandidates={}

    for pid in patient_ids:

        patient_annotations = aapm_id2label[pid]
        # get the patients candidates
        # how did that work?
        candidate_path=aapm_id2candidates_path[pid]

        #load candidates
        candidates=utils.load_pkl(candidate_path)
        neg_candidates=[candidate for candidate in candidates]



        for current_annotation in patient_annotations:
            tmp_candidates = [candidate for candidate in neg_candidates]
            neg_candidates = []
            for candidate in tmp_candidates:
                if np.sqrt(np.sum((current_annotation-candidate)**2)) > 64:
                    neg_candidates.append(candidate)

        print "pid: {}".format(pid)
        print "number of annotations: {}".format(len(patient_annotations))
        print "number of candidates: {}".format(len(candidates))
        print "number of negative candidates: {}".format(len(neg_candidates))
        
        pid2negcandidates[pid]=neg_candidates

        with open(pathfinder.AAPM_CANDIDATES_PATH,'wb') as csvfile:
            writer=csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['id', 'z', 'y', 'x'])
            for (pid,candidate_list) in pid2negcandidates.iteritems():
                for candidate in candidate_list:
                   
                    writer.writerow([pid,candidate[0],candidate[1],candidate[2],0.0])
  


        
if __name__=='__main__':

    aapm_candidates_config = 'aapm_s2_p8a1'

    predictions_dir = utils.get_dir_path('model-predictions', pathfinder.METADATA_PATH)
    aapm_candidates_path = predictions_dir + '/%s' % aapm_candidates_config
    write_aapm_candidates(aapm_candidates_path)


                
