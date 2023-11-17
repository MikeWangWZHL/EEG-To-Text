import os
from glob import glob
import json

print('##############################')
print('start generating ZuCo task1-SR sentiment labels...')


sentiment_labels_task1_csv_path = './dataset/ZuCo/task_materials/sentiment_labels_task1.csv'

sentiment_labels = {}
with open(sentiment_labels_task1_csv_path, 'r') as f:
    for line in f:
        if line.startswith('sentence_id') or line.startswith('#'):
            continue
        else:
            parsed_line = line.split(';')
            # handle edge case:
            if '\";' in line:
                sent_text = line.split('\";')[0].split('\"')[1]
            else:
                sent_text = parsed_line[1]
            label = int(parsed_line[-1].strip())
            sentiment_labels[sent_text] = label

output_dir = f'./dataset/ZuCo/task1-SR/sentiment_labels'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, 'sentiment_labels.json'), 'w') as out:
    json.dump(sentiment_labels,out,indent = 4)
    print('write to ./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json')

