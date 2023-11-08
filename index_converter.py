import json
import os
from tqdm import tqdm
top_n = 2

output_dir = './real_kilt_rb'
topn_dir = output_dir+str(top_n)

if not os.path.exists(topn_dir):
    os.mkdir(topn_dir)

def get_topk(d, n):
    v = sorted(d['vector'].items(),reverse=True)
    d['vector'] = dict(v[:min(n,len(v))])
    return d

for file in tqdm(os.listdir(output_dir)):
    topn_file = topn_dir + '/' + file
    file = output_dir + '/' + file
    with open(topn_file, 'a+') as tf:
        s = ''
        with open(file, 'r') as f:
            for line in f.readlines():
                d = json.loads(line)
                rd = get_topk(d, top_n)
                s += json.dumps(rd) + os.linesep
        tf.write(s)
        

