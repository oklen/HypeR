import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from splade.src.models.transformer_rep import Splade
import pickle
import gc

passage_name = 'kilt_passages_2048_0.pkl'
title_name = 'kilt_w100_title.tsv'
mapping_name = 'mapping_KILT_title.p'
data_dir = '/home/v-zefengcai/models/'


from multiprocessing import Pool

import csv, pickle
tsv_file = open(data_dir + title_name)
read_tsv = csv. reader(tsv_file, delimiter="\t")
titles = [*read_tsv]
# passages = pickle.load(open(data_dir + passage_name, 'rb'))
# mappings = pickle.load(open(data_dir + mapping_name, 'rb'))

print(len(titles))
# print(len(passages))
# print(len(mappings))


real_titles = []
for i in range(1,len(titles)):
    real_titles.append({'id':titles[i][0],'text':titles[i][1],'title':titles[i][2]})


def split_data(data, num, output_dir):
    bs = len(data) // num + 1
    datas = []
    for i in range(num):
        datas.append((output_dir, i, data[i*bs:min((i+1)*bs,len(data))]))
    return datas

def process(x):
    output_dir,cuda_index, real_titles = x
    m_passage = []
    import os
    import numpy as np
    
    def sparse_vector_to_dict(sparse_vec, vocab_id2token, quantization_factor, dummy_token):
        if isinstance(sparse_vec, tuple):
            idx, data = sparse_vec
        else:
            idx = np.nonzero(sparse_vec)[0]
            data = sparse_vec[idx]
        data = np.rint(data * quantization_factor).astype(int)
        dict_sparse = dict()

        for id_token, value_token in zip(idx, data):
            if value_token > 0:
                real_token = vocab_id2token[int(id_token)]
                dict_sparse[real_token] = int(value_token)
        if len(dict_sparse.keys()) == 0:
            dict_sparse[dummy_token] = 1
            # in case of empty doc we fill with "[unused993]" token (just to fill
            # and avoid issues with anserini), in practice happens just a few times ...
        return dict_sparse
    
    from tqdm import tqdm
    import math
    import fcntl
    import json
    bs = 50
    rt = int(math.ceil(len(real_titles) /  bs))
    iter_count = 0
    # gc.collect()
    model_type_or_dir = "naver/splade-cocondenser-ensembledistil"
    model = Splade(model_type_or_dir, agg="max")
#     model.load_state_dict(torch.load('/relevance2-nfs/zefeng/web-t5/outputs/splade-structured_zeroshot,aidayago2,fever,trex,nq,hotpotqa,triviaqa,wow-1-False', map_location='cpu'))
    model.eval()
    model.to('cuda:'+str(cuda_index))
    
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    for index in tqdm(range(1,rt)):
        iter_count += 1
        nw = real_titles[bs*(index-1):min(bs*index,len(real_titles))]
        ps = []
        ids = []
        for w in nw:
            ps.append(w['title']+'. '+w['text'])
            ids.append(w['id'])
        args = tokenizer(ps, return_tensors="pt",truncation=True,padding=True)
        for k in args.keys():
            args[k] = args[k].to('cuda:' + str(cuda_index))
    #     gc.enable()
        with torch.no_grad():
            ret = model(d_kwargs=args)["d_rep"].detach().cpu().squeeze()
    #     gc.disable()
        for i in range(len(ids)):
            query_vec = ret[i]
            sparse_idxs = torch.nonzero(query_vec)
            # print('sparse idxs shape:', sparse_idxs.shape)
            sparse_values = query_vec[sparse_idxs]
            tuple_vec = (sparse_idxs.cpu().numpy(), sparse_values.cpu().numpy())
            dict_sparse = sparse_vector_to_dict(
                tuple_vec, reverse_voc, 100, dummy_token=tokenizer.unk_token)
            context = nw[i]['title'] + '. ' + nw[i]['text']
            m_passage.append({'id':ids[i], 'content':context, 'vector': dict_sparse})
        del ret

    s = ""
    for i,p in enumerate(m_passage):
        s += json.dumps(p) + os.linesep
        if i % 100000 == 0:
            with open(output_dir+'/trained_splade_for_indexing.jsonl' + '-' + str(cuda_index) + '-' + str(i//100000), "a+") as output_file:
                output_file.write(s)
            s = ''
    if len(s) != 0:
        with open(output_dir+'/trained_splade_for_indexing.jsonl' + '-' + str(cuda_index) + '-' + str((len(m_passage)-1)//100000), "a+") as output_file:
            output_file.write(s)

    # for i,p in enumerate(m_passage):
    #     with open(output_dir+'/trained_splade_for_indexing.jsonl' + '-' + str(cuda_index) + '-' + str(i//10000), "a+") as output_file:
    #         output_file.write(json.dumps(p) + os.linesep)

dataset = 'real_kilt'
output_dir = './' + dataset+'_rb'
import os
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

#         os.remove('trained_splade_for_indexing.jsonl')
#     for i in range(1,len(titles)):
#         real_titles.append({'id':titles[i][0],'text':titles[i][1],'title':titles[i][2]})
# for k,v in corpus.items():
#     real_titles.append({'id':k,'text':v['text'],'title':v['title']})

sd = split_data(real_titles, 8, output_dir)
with Pool(8) as p:
    print(p.map(process, sd))
commend = f'/relevance2-nfs/zefeng/anserini/target/appassembler/bin/IndexCollection -collection JsonVectorCollection -input {output_dir}  -index /relevance2-nfs/zefeng/anserini/indexes/lucene-index.beir-{dataset}-passage-distill-splade-zero  -generator DefaultLuceneDocumentGenerator  -threads 92 -impact -pretokenized'
print("{} process done. Run {} to generate coresponding index.".format(dataset,commend))
