import json
import pandas as pd
import os
from IPython import embed


path = 'results/baselines' #'results/hypersearch8_internal' 
drop_keys = ['predict', 'decision', 'params'] 
results = []
for p, _, file in os.walk(path):
    print(p,file)
    if any([x in p for x in ['older_runs', 'demo', 'IB', 'Preprocessed']]):
        continue
    
    for f in file:
        if f == 'results.json':
            f_path = os.path.join(p, f)
            with open(f_path, 'r') as fp:    
                data = json.load(fp)
                keys = data.keys() 
                used_keys = [key for key in keys if not any([word in key for word in drop_keys])]
                used_data = {key:data[key] for key in used_keys}
                
                dataset = p.split('/')[-1].split('_')[0]
                used_data['dataset'] = dataset
                results.append(used_data)
            print(p)
            #for key in data.keys():
            #    if 'val' in key:
            #        print(key)
            #        print(data[key]) 

df = pd.DataFrame(results)
df = df.sort_values(by=['dataset','method'])
embed()
