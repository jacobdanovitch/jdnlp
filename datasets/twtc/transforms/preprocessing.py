"""
Quick one-off script specifically for this dataset.
"""

import pandas as pd
import re

def fix_reports(txt):    
    grades_list = re.findall(r'[A-Za-z]*\: \d+', txt)
    grades = (g.split(': ') for g in grades_list)
    grades = {name: int(val) for (name, val) in grades}

    txt = txt[txt.index(grades_list[-1]) + len(grades_list[-1]):]
    return {'report': txt, **grades}




df = pd.read_csv("../twtc.csv")
df = df[df.label != -1]

mask = df.report.str.startswith('Scouting grades:')
idx = mask.where(mask == True).dropna().index

fixed_vals = df.report[mask].apply(fix_reports)
update_df = pd.DataFrame.from_dict(fixed_vals.values.tolist(), orient='columns').set_index(idx).fillna(0)
df.update(update_df)

df['report'] = df.report.str.encode('ascii', 'ignore').str.decode('ascii')

df.to_csv('../labelled.csv', index=False)


