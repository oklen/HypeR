from kilt import eval_downstream
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", help="Results from the model after trained n epoch", type=str, default=0)

args = parser.parse_args()
path_dir = '/relevance2-nfs/zefeng/web-t5/'
gold_file = 'data/aidayago2-dev-kilt.jsonl'
pred_file = f"outputs/aidayago2-{args.epoch}-dev-kilt.jsonl"
print(eval_downstream.evaluate(path_dir + gold_file,path_dir + pred_file))