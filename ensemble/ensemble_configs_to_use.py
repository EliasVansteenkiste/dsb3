import pathfinder

#############################
# STAGE 1
#############################

FG_CONFIGS = ['fgodin/' + config for config in
              ['dsb_af19lme_mal2_s5_p8a1', 'dsb_af25lme_mal2_s5_p8a1', 'dsb_af4_size6_s5_p8a1',
               'dsb_af5lme_mal2_s5_p8a1', 'dsb_af5_size6_s5_p8a1', 'dsb_af24lme_mal3_s5_p8a1',
               'dsb_af34lme_mal7_s5_p8a1']]

EV_CONFIGS = ['eavsteen/' + config for config in ['dsb_a_eliasq1_mal2_s5_p8a1', 'dsb_a_eliasq10_mal2_s5_p8a1']]

STD_CONFIGS = ['dsb_a04_c3ns2_mse_s5_p8a1', 'dsb_a07_c3ns3_mse_s5_p8a1', 'dsb_a08_c3ns3_mse_s5_p8a1',
               'dsb_a11_m1zm_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1', 'dsb_a_liolme16_c3_s2_p8a1',
               'dsb_a_liolme32_c3_s5_p8a1', 'dsb_a_liox10_c3_s2_p8a1', 'dsb_a_liox11_c3_s5_p8a1',
               'dsb_a_liox12_c3_s2_p8a1',
               'dsb_af25lmelr10-2_mal2_s5_p8a1', 'dsb_af25lmelr10-1_mal2_s5_p8a1', 'dsb_a_eliasz1_c3_s5_p8a1',
               'dsb_a_liox13_c3_s2_p8a1', 'dsb_a_liox14_c3_s2_p8a1', 'dsb_a_liox15_c3_s2_p8a1',
               'dsb_a_liox6_c3_s2_p8a1',
               'dsb_a_liox7_c3_s2_p8a1', 'dsb_a_liox8_c3_s2_p8a1', 'dsb_a_liolunalme16_c3_s2_p8a1',
               'dsb_a_lionoclip_c3_s5_p8a1', 'dsb_a_liomse_c3_s5_p8a1', 'dsb_af25lmeo0_s5_p8a1',
               'dsb_a_liomseresume_c3_s5_p8a1', 'dsb_af25lmelr10-3_mal2_s5_p8a1', 'dsb_a_liomix_c3_s5_p8a1',
               'dsb_a_liomselunaresume_c3_s5_p8a1']

GOOD_CONFIGS = ['fgodin/dsb_af25lme_mal2_s5_p8a1', 'dsb_af25lmeaapm_mal2_s5_p8a1', 'dsb_a_liolme16_c3_s2_p8a1',
                'dsb_a_liolme32_c3_s5_p8a1', 'dsb_af25lmelr10-2_mal2_s5_p8a1', 'dsb_a_liox13_c3_s2_p8a1',
                'dsb_af25lmeo0_s5_p8a1', 'dsb_af25lmelr10-3_mal2_s5_p8a1', 'eavsteen/dsb_a_eliasq1_mal2_s5_p8a1']

#############################
# STAGE 2
#############################

# should be the _spl
STAGE_2_CONFIGS = ['example_config_preds_spl'
                   ]


def get_spl_configs():
    if pathfinder.STAGE == 1:
        return FG_CONFIGS + STD_CONFIGS + EV_CONFIGS
    else:
        return STAGE_2_CONFIGS


def get_allin_configs():
    if pathfinder.STAGE == 1:
        return FG_CONFIGS + STD_CONFIGS + EV_CONFIGS
    else:
        return STAGE_2_CONFIGS
