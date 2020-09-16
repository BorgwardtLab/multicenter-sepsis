#copied from Physionet2019 Dataset Class 
ts_columns = [
        'hr', 'o2sat', 'temp', 'sbp', 'map', 'dbp', 'resp',
        'etco2', 'be', 'bicar', 'fio2', 'ph', 'pco2', 'ast', 
        'bun', 'alp', 'ca', 'cl', 'crea', 'bili_dir', 'glu', 
        'lact', 'mg', 'phos', 'k', 'bili', 'tri', 'hct', 'hgb', 
        'ptt', 'wbc', 'fgn', 'plt', 'alb', 'alt', 'basos', 
        'bnd', 'cai', 'ck', 'ckmb', 'crp', 'eos', 'esr', 'hbco',
        'inr_pt', 'lymph', 'mch', 'mchc', 'mcv', 'methb', 'na', 
        'neut', 'po2', 'pt', 'rbc', 'rdw', 'tco2', 'tnt', 'sirs', 
        'news', 'mews'
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
        'neut', 'po2', 'pt', 'rbc', 'rdw', 'tco2', 'tnt', 'sirs', 
        'news', 'mews', 
        #Derived:
        'ShockIndex', 'bun/cr', 'po2/fio2', 'SOFA', 'SOFA_deterioration',  
        'qSOFA', 'SepticShock', 'MEWS', 'SIRS'
    ]
#    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess',
#    'HCO3', 'FiO2', 'pH', 'PaCO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium',
#    'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
#    'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI',
#    'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'ShockIndex',
#    'BUN/CR', 'O2Sat/FiO2', 'SOFA', 'SOFA_deterioration', 'sofa_max_24hrs',
#    'qSOFA', 'SepticShock', 'MEWS', 'SIRS'
#]

columns_not_to_normalize = [
    'sex', 'stay_time', 'sep3'
]
