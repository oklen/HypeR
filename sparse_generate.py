import os
import pickle
import time
import argparse
import subprocess

from data import KiltDataset, seq2seq_to_kilt, dataset_config, remove_preds, dataset_task_map

def load_list_from_file(file_path, encoding="utf-8"):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding=encoding) as fp:
            for line in fp:
                data.append(line.strip())
    return data

ctx_file = '/relevance2-nfs/zefeng/models/kilt_w100_title.tsv.pkl'
# ctx_file = '/relevance2-nfs/zefeng/models/psgs_w100.tsv.pkl'
with open(ctx_file, 'rb') as f:
    all_passages = pickle.load(f)
KILT_mapping_path = '/relevance2-nfs/zefeng/models/mapping_KILT_title.p'
# KILT_mapping_path =  '/home/v-zefengcai/models/300_KILT_mapping.pkl'
KILT_mapping = pickle.load(open(KILT_mapping_path, "rb"))

def find_and_process(dataset, output_dir):
    for file in os.listdir(output_dir):
        if 'wait-to-process' not in file or dataset not in file:
            continue
        try:
            os.remove(output_dir + file)
        except:
            print(f"{output_dir + file} have been removed.")
            continue
        print(f"Begin Process {file}")
        vec_path, epoch_count, _ = file.split('##')
        evaluate(dataset, output_dir + vec_path, epoch_count)
        print(f"Evaluate done: {file}")


def evaluate(dataset, vec_path, epoch_count):
    # commend = f"/relevance2-nfs/zefeng/anserini/target/appassembler/bin/SearchCollection -hits 100 -parallelism 95 \
    #             -index /relevance2-nfs/zefeng/anserini/indexes/lucene-index.msmarco-passage-distill-splade-max \
    #             -topicreader TsvString -topics {vec_path} \
    #             -output {vec_path}-results-{dataset} -format trec \
    #             -impact -pretokenized >> anserini.logs"
    commend = f"/relevance2-nfs/zefeng/anserini/target/appassembler/bin/SearchCollection -hits 100 -parallelism 95 \
                -index /relevance2-nfs/zefeng/anserini/indexes/lucene-index.msmarco-passage-distill-splade-max \
                -topicreader TsvString -topics {vec_path} \
                -output {vec_path}-results-{dataset} -format trec \
                -impact -pretokenized >> anserini.log"

    # commend = f"/relevance2-nfs/zefeng/anserini/target/appassembler/bin/SearchCollection -hits 100 -parallelism 95 \
    #             -index /relevance2-nfs/zefeng/anserini/indexes/lucene-index.msmarco-passage-distill-splade-mt-zero \
    #             -topicreader TsvString -topics {vec_path} \
    #             -output {vec_path}-results-{dataset} -format trec \
    #             -impact -pretokenized >> anserini.log"
    os.system(commend)

    lines = load_list_from_file(vec_path+'-results-'+dataset)

    # with open(self.hparams.last_sparse_output_dev) as f:
    #     lines = load_list_from_file(f)
    query_id2provance = {}
    for line_info in lines:
        sp = line_info.split(' ')
        if len(sp) == 6:
            qid, _, pid, rank, score, _method = sp
        else:
            qid1, qid2, _, pid, rank, score, _method = sp
            qid = " ".join([qid1, qid2])
        if query_id2provance.get(qid) is None:
            query_id2provance[qid] = []
        t_passage = all_passages[pid]
        wiki_id = KILT_mapping.get(t_passage[1])
        if wiki_id is None:
            wiki_id = 0
            print("Cound not find title in passage:", t_passage[1])

        query_id2provance[qid].append(
        {
            "wikipedia_title": t_passage[1],
            "wikipedia_id": wiki_id,
            "text": t_passage[0],
        }
        )

    preds = []
    ids = []
    sources = []
    targets, provance = [], []
    for id, p in query_id2provance.items():
        ids.append(id)
        provance.append(p)
    sources = [''] * len(provance)
    preds = [''] * len(provance)
    gen_file_path = '/relevance2-nfs/zefeng/old_try/web-t5/outputs/' + dataset + '-' + str(epoch_count) + '-auto-sparse-dev-kilt.jsonl'
    # gen_file_path = '/relevance2-nfs/zefeng/web-t5/outputs/' + dataset + '-' + str(epoch_count) + '-auto-sparse-dev-kilt.json'
    if os.path.exists(gen_file_path):
        os.remove(gen_file_path)

    answer_path = '/relevance2-nfs/zefeng/web-t5/data/' + dataset + '-dev-kilt.jsonl'
    # answer_path = '/relevance2-nfs/zefeng/web-t5/new_data/' + dataset + '-dev-multikilt.json'
    data_file = seq2seq_to_kilt(ids, sources, preds, '/relevance2-nfs/zefeng/old_try/web-t5/outputs',
            dataset + f"-{epoch_count}-auto-sparse", 'dev', provance)
    commond = f'python /relevance2-nfs/zefeng/old_try/web-t5/eval_retrieval.py {gen_file_path} {answer_path}'
    print(subprocess.check_output(['python','/relevance2-nfs/zefeng/old_try/web-t5/eval_retrieval.py', data_file, answer_path], text=True))
    # print(os.system(commend))
    try:
        os.remove(vec_path)
    except:
        pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Choice dataset for evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="/relevance2-nfs/zefeng/old_try/web-t5/outputs/",
        help="Select output dir",
    )
    args = parser.parse_args()
    print(f"Working in {args.output_dir}")
    while True:
        find_and_process(args.dataset, args.output_dir)
        time.sleep(5)

#     evaluate(args.gold, args.guess, args.ks, args.rank_keys)