import os
import pickle
import time
import argparse
import subprocess
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler

from data import KiltDataset, seq2seq_to_kilt, dataset_config, remove_preds, dataset_task_map

def load_list_from_file(file_path, encoding="utf-8"):
    data = []
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding=encoding) as fp:
            for line in fp:
                data.append(line.strip())
    return data


# beir_dataset = {'msmarco','nfcorpus','bioasq','nq','hotpotqa','fiqa','arguana','webis-touche2020','cqadupstack','quora','dbpedia-entity','scidocs','fever','climate-fever','scifact','trec-covid-v2'}
beir_dataset = {'msmarco', 'nfcorpus','bioasq','hotpotqa','fiqa','arguana','webis-touche2020','quora','dbpedia-entity','scidocs','fever','climate-fever','scifact','trec-covid-v2'}

def find_and_process(dataset, output_dir):
    for file in os.listdir(output_dir):
        if 'wait-to-process' not in file or not any(i in file for i in beir_dataset):
            continue
        for i in beir_dataset:
            if i in file:
                dataset = i
                break
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
                    -index /relevance2-nfs/zefeng/anserini/indexes/lucene-index.beir-{dataset}-passage-distill-splade-zero \
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
    if True:
        results = {}
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        # out_dir = os.path.join("./", "datasets")
        out_dir = os.path.join("/relevance2-nfs/zefeng/old_try/web-t5/", "datasets")
        data_path = util.download_and_unzip(url, out_dir)

        #### Provide the data_path where scifact has been downloaded and unzipped
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        # corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
        # print(qrels)

        retriever = EvaluateRetrieval()
        for line_info in lines:
            sp = line_info.split(' ')
            if len(sp) == 6:
                qid, _, pid, rank, score, _method = sp
            else:
                qid1, qid2, _, pid, rank, score, _method = sp
                qid = " ".join([qid1, qid2])
            if results.get(qid) is None:
                results[qid] = {}
            results[qid][pid] = float(score)
        #     print(qid, pid, score)

        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,5,10,20])
        print('ndcg:',ndcg)
        print('recall:',recall)
        print('precision:', precision)

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