# beir_dataset = {'msmarco','trec-covid','nfcorpus','bioasq','fiqa','arguana','webis-touche2020','cqadupstack','quora','dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
# beir_dataset = {'nfcorpus','fiqa','arguana','webis-touche2020','quora','dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
# beir_dataset = {'nfcorpus','fiqa','arguana','webis-touche2020','quora','dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
beir_dataset = {'nfcorpus','fiqa','arguana','webis-touche2020',,'dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
# beir_dataset = {'msmarco'}

with open('./test_beir.sh','w') as f:
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-1-True-Truejs=0#01_expand=10#n_tokens=50-question_encoder'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truejs=0#01_expand=5#n_tokens=50-question_encoder'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truepure_prompt-question_encoder'

    # model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-9-True-Truejs=0.1#expand=30#n_tokens=5'
    # model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-0-True-Truejs=0.1#expand=5#n_tokens=50'
    # model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-5-True-Truejs=0.1#expand=5#n_tokens=1#fix_temp#2'
    # model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-0-True-Truejs=0.1#expand=5#n_tokens=1#fix_temp#2#100h'
    share_expand = '5'
    model_file = 'splade-fever,nq,triviaqa,hotpotqa-1-True-Trueb1'

    for dataset in beir_dataset:
        mpt_mode = 'False'
        use_prompt = 'True'
        use_v2 = 'True'
        n_tokens = '1'
        fixed_temp_step = '40000'
        commend = f"../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False dataset={dataset} local_process=False \
        num_workers=1 use_prompt={use_prompt} mpt_mode={mpt_mode} n_gpu=1 use_v2={use_v2}\
        n_tokens={n_tokens} fixed_temp_step={fixed_temp_step} share_expand={share_expand} do_train=False load_question_name=\\\'{model_file}\\\' \
        customer_keys=b2"
        f.write(commend + '\n')
