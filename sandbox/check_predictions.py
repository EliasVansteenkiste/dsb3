import utils

def check_ira(prediction_file):
    d = utils.load_pkl(prediction_file)
    print d.keys()
    print d['predictions'][0].keys()
    for i in xrange(0, 1140):
        print d['predictions'][i]['patient']
        print len(d['predictions'][i]['systole'])
        print len(d['predictions'][i]['diastole'])
        print '--------------------------'

# mtd_name = 'gauss_roi10_maxout_seqshift_96-geit-20160308-130617.pkl'
#mtd_name = 'meta_gauss_roi10_maxout-geit-20160309-045347.pkl'
# mtd_name = 'je_ss_jonisc80small_360_gauss_longer_augzoombright.pkl'
# mtd_name = 'gauss_roi10_maxout_seqshift_96-geit-20160308-130617.pkl'
# mtd_name = 'gauss_roi10_maxout-geit-20160308-130407.pkl'
# mtd_name = 'gauss_roi10_big_leaky_after_seqshift-geit-20160309-104330.pkl'
# check_jeroen(mtd_name)
mtd_name = 'ira_configurations.meta_gauss_roi_zoom_big.pkl'
check_ira(mtd_name)