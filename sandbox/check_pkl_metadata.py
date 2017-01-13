import utils

def check_ira(mtd_name):
    mtd = utils.load_pkl(mtd_name)
    print mtd.keys()

    for lt, lv, lcv in zip(mtd['losses_eval_valid'], mtd['losses_train'], mtd['crps_eval_valid']):
        print lt, lv, lcv

    print mtd['chunks_since_start']
    print mtd['configuration']
    print mtd.get('subconfiguration', None)


def check_jeroen(mtd_name):
    mtd = utils.load_pkl(mtd_name)
    print mtd.keys()

    for lt, lv, lct, lcv in zip(mtd['losses_eval_valid'], mtd['losses_eval_train'], mtd['losses_eval_train_kaggle'],
                                mtd['losses_eval_valid_kaggle']):
        print lt, lv, lct, lcv

    print mtd['chunks_since_start']
    print mtd['configuration_file']
    print mtd.get('subconfiguration', None)

# mtd_name = 'gauss_roi10_maxout_seqshift_96-geit-20160308-130617.pkl'
#mtd_name = 'meta_gauss_roi10_maxout-geit-20160309-045347.pkl'
# mtd_name = 'je_ss_jonisc80small_360_gauss_longer_augzoombright.pkl'
# mtd_name = 'gauss_roi10_maxout_seqshift_96-geit-20160308-130617.pkl'
# mtd_name = 'gauss_roi10_maxout-geit-20160308-130407.pkl'
# mtd_name = 'gauss_roi10_big_leaky_after_seqshift-geit-20160309-104330.pkl'
mtd_name = 'gauss_roi10_zoom_mask_leaky_after-kip-20160311-225259.pkl'
# check_jeroen(mtd_name)
check_ira(mtd_name)