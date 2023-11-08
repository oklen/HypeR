# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import configparser
import fcntl
import gzip
import json
import os
import pathlib
import random
import copy

import torch.utils.data
#from transformers.tokenization_utils import trim_batch

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

dataset_task_map = {'nq': "Question Answering", "aidayago2": "Entity Linking", "cweb": "Entity Linking",
                    "fever": "Fact Checking", "hotpotqa": "Question Answering",
                    "triviaqa": "Question Answering", "wned": "Entity Linking", "wow": "Dialogue",
                    "zeroshot": "Relation Extraction", "trex":"Slot Filling","structured_zeroshot":"Relation Extraction", "eli5":"Question Answering"}

# Remove nq,fever,hotpotqa for overlap!
beir_dataset = {'msmarco','trec-covid','nfcorpus','bioasq','fiqa','arguana','webis-touche2020','cqadupstack','quora','dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
# beir_dataset = {'msmarco','trec-covid','nfcorpus','bioasq','nq','hotpotqa','fiqa','arguana','webis-touche2020','cqadupstack','quora','dbpedia-entity','scidocs','fever','climate-fever','scifact'}

dataset_config = configparser.ConfigParser()
# location = os.path.join(pathlib.Path(__file__).parent, 'config_file')
location = "/home/czf/Code/KILT/kilt/reader/t5/config_file"
dataset_config.read(location)


def encode_seq(tokenizer, seqs, max_length, out_dir, dataset, side='source', type_path='train', pad_to_max_length=True,
               return_tensors="pt", is_query=False, prompt_text=None):
    examples = []
    lengths = []
    if is_query is False:
        output_file = os.path.join(out_dir, dataset + "-" + type_path + "-" + side + ".encoded")
    else:
        type_path = type_path + "-query"
        output_file = os.path.join(out_dir, dataset + "-" + type_path + "-" + side + ".encoded")

    # if side == 'source':
    #     p_text = prompt_text[dataset_task_map[dataset]]

    with open(output_file, "w") as f_out:
        texts = []
        for text in seqs:
            if dataset_task_map[dataset] == 'Entity Linking' and side == 'source':
                length = int(int(max_length) / 2)
                mention_start = text.find('[START_ENT]')
                mention_end = text.find('[END_ENT]')
                left = text[0:mention_start]
                right = text[mention_end + len('[END_ENT]'):]

                left_ids = tokenizer.encode(left)
                right_ids = tokenizer.encode(right)
                left = tokenizer.decode(left_ids[max(0, len(left_ids) - length):len(left_ids)])
                right = tokenizer.decode(right_ids[0:min(len(right_ids), length)])
                text = left + ' ' + text[mention_start:mention_end] + '[END_ENT] ' + right
                # Avoid add specail tokens
                text.replace('[START_ENT]','[unused200]')
                text.replace('[END_ENT]','[unused201]')


            if dataset == 'wow' and side == 'source':
                text = text.replace('\n', '[SEP]')

            if dataset == 'fever' and side == 'target':
                if text == "REFUTES":
                    text = "<REFUTES>"
                if text == "SUPPORTS":
                    text = "<SUPPORTS>"

            if not is_query:
                text = text + '</s>' if side == 'target' else \
                    dataset_task_map[dataset] + ": " + text

            # Add prompt for source input
            # if type == 'source':
            #     text = p_text + text
            texts.append(text)

        # if dataset == 'wow' and side == 'source':
        #     tokenized = tokenizer.batch_encode_plus(
        #         texts, add_special_tokens=True, max_length=max_length, pad_to_max_length='left',
        #         # return_tensors=return_tensors,
        #     )
        # elif side == 'source':
        #     tokenized = tokenizer.batch_encode_plus(
        #         texts, add_special_tokens=True, max_length=max_length, pad_to_max_length=pad_to_max_length,
        #         # return_tensors=return_tensors,
        #     )
        # else:
        #     tokenized = tokenizer.batch_encode_plus(
        #         texts, add_special_tokens=False, max_length=max_length, pad_to_max_length=pad_to_max_length,
        #         truncation=True,
        #         # return_tensors=return_tensors,
        #     )

        # if not is_query:
        #     for input in tokenized["input_ids"]:
        #         tokens = tokenizer.convert_ids_to_tokens(input)
        #         f_out.write(' | '.join(tokens) + "\n")
        # else:
        #     for input in texts:
        #         f_out.write(input + "\n")

        #lengths.append(tokenized["input_ids"].size()[1])

    # Keep texts for query + knowledge
    if is_query:
        return texts, None
    else:
        return tokenized

from transformers import AutoTokenizer

def filter(answer, ng):
    real_ng = []
    for n in ng:
        if answer not in n['text']:
            real_ng.append(n)
    return real_ng

class KiltDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            dataset,
            type_path,
            max_source_length,
            max_target_length,
            output_dir,
            config=None,
    ):
        print("Load {}.".format(dataset))
        super().__init__()
        # self.tokenizer = tokenizer
        self.dpr_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.config = config
        self.negative_samples = None
        self.answer_mask = None

        Task_type = ['Question Answering', 'Entity Linking', 'Fact Checking', 'Dialogue', 'Relation Extraction','Slot Filling']
        prompt_tokens = []
        self.prompt_text = {}
        # for t in Task_type:
        #     tmp_prompt = []
        #     tmp_text = ""
        #     for i in range(config.soft_prompt_length):
        #         tmp_prompt.append(t+'#'+str(i))
        #         tmp_text += tmp_prompt[-1]
        #     if config.use_prompt:
        #         self.prompt_text[t] = tmp_text
        #     else:
        #         self.prompt_text[t] = ''
        #     prompt_tokens.extend(tmp_prompt)
        
        # self.tokenizer.add_tokens(prompt_tokens)

        # id2gold_wiki_id = {}



        # self.ids, raw_sources, raw_targets, self.id_targets = nq_jsonl_to_tsv(data_dir, type_path)
        if dataset not in beir_dataset:
            self.ids, raw_sources, raw_targets, self.id_targets, self.provenance, gold_page = kilt_to_seq2seq(data_dir, dataset, type_path)
            self.data_name = dataset
        else:
            dataset_task_map[dataset] = 'Question Answering'
            self.ids, raw_sources, raw_targets, self.id_targets, self.provenance, gold_page = beir_to_seq2seq(data_dir, dataset, type_path)
            self.data_name = dataset

        # for id, p in zip(self.ids, self.provenance):
        #     if id2gold_wiki_id.get(id) is None:
        #         id2gold_wiki_id[id] = set()
        #     # print(p)
        #     id2gold_wiki_id[id].add(p['wikipedia_id'])

        # self.source = encode_seq(tokenizer, raw_sources, max_source_length, output_dir, dataset, 'source', type_path, prompt_text=self.prompt_text)
        
        if config.dense_model:
            negative_samples_dir = config.negative_samples_dir + dataset + '-train-kilt.jsonl'
        else:
            negative_samples_dir = config.sparse_negative_samples_dir + dataset + '-train-kilt.jsonl'

        # negative_samples_dir: /relevance2-nfs/zefeng/web-t5/predictions/bm25/aidayago2-train-kilt.jsonl
        self.query_txt, self.dpr_source = encode_seq(self.dpr_tokenizer, raw_sources, 512, output_dir, dataset, 'source', type_path, is_query=True, prompt_text=self.prompt_text)
        self.id_to_ans = {}

        if config and config.retrieve_mode and type_path == 'train' and config.do_train:
            self.id_to_negative_samples = {}
            last_provenance = None
            with open(negative_samples_dir) as f:
                lines = f.readlines()
                for line in lines:
                    js = json.loads(line)
                    p = js['output'][0]['provenance']
                    for i in range(len(p)):
                        try:
                            p[i]['title'] = p[i]['wikipedia_title']
                        except:
                            p[i]['title'] = p[i]['text'].split('.')[0]
                    # last_provenance = self.id_to_negative_samples[js['id']] = p[:min(100, config.ng_count)]
                    # last_provenance = self.id_to_negative_samples[js['id']] = random.sample(p, min(100, config.ng_count))
                    self.id_to_negative_samples[js['id']] = p

            if len(gold_page) < len(self.ids):
                gold_page = gold_page + (len(self.ids) - len(gold_page)) * [set()]
            self.negative_samples = []
            self.answer_mask = None
            # self.answer_mask = []
            empty_count = 0
            for tmp_id,answer, one_gold_page in zip(self.ids, raw_targets, gold_page):
                # try:
                # rgs = filter(answer, self.id_to_negative_samples[tmp_id][-90:])

                # if len(rgs) >= config.ng_count:
                #     self.negative_samples.append(rgs)
                # else:
                #     self.negative_samples.append(self.id_to_negative_samples[tmp_id][-2 * config.ng_count:])
                ng = self.id_to_negative_samples[tmp_id]
                ngs = []
                for n in ng:
                    if not config.hparams.beir_mode and n['wikipedia_title'] in one_gold_page:
                        continue
                    ngs.append(n)

                if len(ngs) >= config.ng_count:
                    # ngs = self.id_to_negative_samples[tmp_id][-config.ng_count:]
                    ngs = random.sample(self.id_to_negative_samples[tmp_id], config.ng_count)
                else:
                    pass
                    # ngs = ngs[:2 * config.ng_count]
                    # ngs

                # if len(ngs) < config.ng_count:
                #     print(dataset)
                #     exit(0)
                self.negative_samples.append(ngs)

                # assert len(self.negative_samples) >= config.ng_count

                # ans_mask = []
                # for ng in self.negative_samples[-1]:
                #     if ng['wikipedia_id'] in self.id2gold_wiki_id[tmp_id]:
                #         ans_mask.append(1)
                #     else:
                #         ans_mask.append(0)
                # ans_mask.append(1)
                # self.answer_mask.append(ans_mask)
            assert len(self.negative_samples) == len(self.query_txt)
        # self.id_map = [*range(len(self.ids))]
        # random.shuffle(self.id_map)
        # only use half data of trex
        # if dataset == 'trex':
        #     ls = len(self.ids) // 2
        #     self.ids = self.ids[:ls]
        #     self.query_txt = self.query_txt[:ls]
        #     self.negative_samples = self.negative_samples[:ls]
        #     self.provenance = self.provenance[:ls]

        # if 'test' not in type_path:
        #     self.target = encode_seq(tokenizer, raw_targets, max_target_length, output_dir, dataset, 'target', type_path, prompt_text=self.prompt_text)
        # else:
        #     # Dummpy target
        #     self.target = self.source
        # In few shot setting we clip the data to specificed size
        if type_path == 'train' and config.few_shot_data_cnt != -1:
            select_id = random.sample([*range(len(self.query_txt))], config.few_shot_data_cnt)
            self.query_txt = [self.query_txt[i] for i in select_id]
            self.ids = [self.ids[i] for i in select_id]
            self.negative_samples = [self.negative_samples[i] for i in select_id]
            self.provenance = [self.provenance[i] for i in select_id]

    def __len__(self):

        return len(self.ids)

    def __getitem__(self, index):

        # source_ids = self.source["input_ids"][index]
        # dpr_source_ids = self.dpr_source["input_ids"][index].squeeze()
        # dpr_masks = self.dpr_source["attention_mask"][index].squeeze()

        # target_ids = torch.tensor(self.target["input_ids"][index]).squeeze()
        # src_mask = torch.tensor(self.source["attention_mask"][index]).squeeze()

        query_txt = self.query_txt[index]
        q_id = self.ids[index]

        # ret = {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "id": q_id, "query_txt": query_txt,}
        ret = {"id": q_id, "query_txt": query_txt,"dataset": self.data_name}
        if self.negative_samples is not None:
            ret["negative_samples"] = self.negative_samples[index]
        if self.provenance is not None:
            ret["gold_provenance"] = self.provenance[index]
        if self.answer_mask is not None:
            ret["answer_mask"] = self.answer_mask[index]

        return ret

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        target_ids = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, target_ids


def kilt_to_seq2seq(data_dir, dataset, type_path):
    data_file = pathlib.Path(os.path.join(data_dir, dataset + '-' + type_path + "-kilt.jsonl"))
    sources = []
    targets = []
    ids = []
    id_targets = {}
    gold_provenance = []
    pargraph_ids = []
    gold_pages = []
    if not data_file.exists():
        raise f"File not exists: {data_file}"
        return ids, sources, targets
    last_provenance = {}

    with open(data_file, "r") as f:
        for line in f.readlines():
            qa = json.loads(line)
            q_id = qa['id']
            question = qa['input']
            output = qa.get('output')

            if output is None or len(output) == 0:
                sources.append(question)
                ids.append(q_id)
                gold_provenance.append(last_provenance)
                continue
            answers = set()
            id_targets[q_id] = []
            gold_page = set()
            for out in output:
                if 'provenance' in out:
                    for p in out['provenance']:
                        gold_page.add(p['title'])

            for out in output:
                if 'answer' not in out.keys():
                    continue
                answer = out['answer']
                if answer in answers:
                    continue
                answers.add(answer)
                id_targets[q_id].append(answer)

                if type_path == 'test':
                    sources.append(question)
                    targets.append(answers.pop())
                    ids.append(q_id)
                    gold_provenance.append(last_provenance)
                    break
                else:
                    if out.get('provenance') is not None:
                        if type_path == 'dev':
                            # When dev, only repeat use provenance
                            last_provenance = out['provenance'][0]
                            gold_provenance.append(out['provenance'][0])
                            sources.append(question)
                            targets.append(answer)
                            ids.append(q_id)
                            break

                        for p in out['provenance']:
                            # Remove Specific answer
                            # if ':::' in p['text']:
                            #     continue
                            last_provenance = p
                            gold_provenance.append(p)
                            sources.append(question)
                            targets.append(answer)
                            ids.append(q_id)
                            gold_pages.append(gold_page)
                            # break
                    elif type_path != 'train':
                        sources.append(question)
                        targets.append(answer)
                        ids.append(q_id)
                        gold_provenance.append(last_provenance)
                # break
        return ids, sources, targets, id_targets, gold_provenance, gold_pages

# Only support test on beir
def beir_to_seq2seq(data_dir, dataset, type_path):
    # data_file = pathlib.Path(os.path.join(data_dir, dataset + '-' + type_path + "-kilt.jsonl"))
    sources = []
    targets = None
    ids = []
    id_targets = None
    gold_provenance = None
    pargraph_ids = None
    gold_pages = None
    # if not data_file.exists():
    #     raise f"File not exists: {data_file}"
    #     return ids, sources, targets
    from beir import util, LoggingHandler
    from beir.retrieval import models
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path("./").parent.absolute(), "datasets")
    out_dir = os.path.join("/relevance2-nfs/zefeng/old_try/web-t5/", "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    #### Provide the data_path where scifact has been downloaded and unzipped
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")


    for id, query in queries.items():
        ids.append(id)
        sources.append(query)

    return ids, sources, targets, id_targets, gold_provenance, gold_pages

def remove_preds(output_dir, dataset, type_path, predicted_provance = None):
    data_file = os.path.join(output_dir, dataset + '-' + type_path + "-kilt.jsonl")
    if os.path.exists(data_file):
        try:
            os.remove(data_file)
        except:
            pass

def seq2seq_to_kilt(ids, sources, targets, output_dir, dataset, type_path, predicted_provance = None):
    data_file = os.path.join(output_dir, dataset + '-' + type_path + "-kilt.jsonl")

    with open(data_file, "a+") as output_file:
        data = []
        if predicted_provance is None:
            for q_id, s, t in zip(ids, sources, targets):
                qa = {"id": q_id, 'input': s, 'output': []}
                a = {'answer': t, 'provenance': []}
                qa['output'].append(a)
                data.append(json.dumps(qa))
        else:
            for q_id, s, t, p in zip(ids, sources, targets, predicted_provance):
                qa = {"id": q_id, 'input': s, 'output': []}
                a = {'answer': t, 'provenance': p}
                # print(p)
                qa['output'].append(a)
                data.append(json.dumps(qa))
        fcntl.flock(output_file, fcntl.LOCK_EX)
        if os.stat(data_file).st_size > 0:
            output_file.write('\n')
        output_file.write('\n'.join(data))
        fcntl.flock(output_file, fcntl.LOCK_UN)
    print("kilt_dump done:",data_file)
    return data_file


def nq_jsonl_to_tsv(data_dir, type_path):
    def extract_answer(answer_tokens, span):
        """Reconstruct answer from token span and remove extra spaces."""
        start, end = span["start_token"], span["end_token"]
        ans = " ".join(answer_tokens[start:end])
        # Remove incorrect spacing around punctuation.
        ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        ans = ans.replace("( ", "(").replace(" )", ")")
        ans = ans.replace("`` ", "\"").replace(" ''", "\"")
        ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
        return ans

    count = 0
    ids = []
    sources = []
    targets = []
    id_targets = {}
    in_fname = data_dir + '/' + type_path + '.jsonl.gz'

    for line in gzip.open(in_fname, "rb"):
        ex = json.loads(line)

        # Remove any examples with more than one answer.

        # Questions in NQ do not include a question mark.
        q_id = ex['annotations'][0]['annotation_id']
        question = ex["question_text"] + "?"
        answers = []
        for answer_span in ex['annotations'][0]['short_answers']:
            tokens = []
            # Handle the two document formats in NQ (tokens or text).
            if "document_tokens" in ex:
                tokens = [t["token"] for t in ex["document_tokens"]]
            elif "document_text" in ex:
                tokens = ex["document_text"].split(" ")
            answer = extract_answer(tokens, answer_span)
            # Write this line as <question>\t<answer>
            sources.append(question)
            targets.append(answer)
            answers.append(answer)
            ids.append(q_id)
        id_targets[q_id] = answers
        count += 1

    return ids, sources, targets, id_targets


if __name__ == "__main__":
    pass
