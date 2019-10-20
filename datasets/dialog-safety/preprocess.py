import json
import pandas as pd


def save_data(data, key): 
    df = pd.DataFrame(data[key])[['text', 'labels']].rename(columns={'labels': 'label'}) 
    df['label'] = df['label'].apply(lambda x: x[0].replace('__', '')) 
    df.to_json(f'datasets/dialog-safety/{key}.json', orient='records', lines=True)


if __name__ == "__main__":
    import sys
    
    data = json.load(sys.argv[1])
    for key in data.keys():
        save_data(data, key)