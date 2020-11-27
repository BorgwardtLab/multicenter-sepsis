# columns to drop and ignore completely (e.g. interventions that should not affect the model)
colums_to_drop = ['abx', 'ins', 'rass']

baseline_cols = ['sirs', 'news', 'mews', 'gcs', 'qsofa', 
        'sofa_cardio', 'sofa_cns', 'sofa_coag', 'sofa_liver',
        'sofa_renal', 'sofa_resp', 'sofa']


# raw measurements
ts_columns = [
        'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
        'etco2', 'be', 'bicar', 'fio2', 'ph', 'pco2', 'ast', 
        'bun', 'alp', 'ca', 'cl', 'crea', 'bili_dir', 'glu', 
        'lact', 'mg', 'phos', 'k', 'bili', 'tri', 'hct', 'hgb', 
        'ptt', 'wbc', 'fgn', 'plt', 'alb', 'alt', 'basos', 
        'bnd', 'cai', 'ck', 'ckmb', 'crp', 'eos', 'esr', 'hbco',
        'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'methb', 'na', 
        'neut', 'po2', 'pt', 'rbc', 'rdw', 'tco2', 'tnt', 
        'vaso_ind', 'vent_ind', 'urine24'
    ]

#adding derived features (which are still time series -- not yet features capturing temporal information!)
extended_ts_columns = [
        'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
        'etco2', 'be', 'bicar', 'fio2', 'ph', 'pco2', 'ast', 
        'bun', 'alp', 'ca', 'cl', 'crea', 'bili_dir', 'glu', 
        'lact', 'mg', 'phos', 'k', 'bili', 'tri', 'hct', 'hgb', 
        'ptt', 'wbc', 'fgn', 'plt', 'alb', 'alt', 'basos', 
        'bnd', 'cai', 'ck', 'ckmb', 'crp', 'eos', 'esr', 'hbco',
        'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'methb', 'na', 
        'neut', 'po2', 'pt', 'rbc', 'rdw', 'tco2', 'tnt', 
        'vaso_ind', 'vent_ind', 'urine24',

        #Derived (only partially observed scores/features)
        'ShockIndex', 'bun/cr', 'po2/fio2', 'SOFA', 'SOFA_deterioration',  
        'qSOFA', 'SepticShock', 'MEWS', 'SIRS'
    ]

static_columns = ['age', 'sex', 'weight', 'height']

columns_not_to_normalize = [ 'sex', 'time', 'sep3' ]  + baseline_cols
#at the beginning of preprocessing, stay_time is mapped to 'time' 
