# beir_dataset = {'msmarco','trec-covid','nfcorpus','bioasq','fiqa','arguana','webis-touche2020','cqadupstack','quora','dbpedia-entity','scidocs','climate-fever','scifact','trec-covid-v2'}
kilt_dataset = 'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow'.split(',')

with open('./test_kilt.sh','w') as f:
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-1-True-Truejs=0#01_expand=10#n_tokens=50-question_encoder'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truejs=0#01_expand=5#n_tokens=50-question_encoder'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truepure_prompt-question_encoder'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truejs=0.01#expand=10#n_tokens=50#for_beir'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-4-True-Truejs=0.01#expand=10#n_tokens=100#for_beir'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-3-True-Truejs=0.01#expand=10#n_tokens=5#for_beir'
    # model_file = 'splade-aidayago2,nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truejs=0.01#expand=10#n_tokens=20#for_beir'
    # model_file = 'splade-nq,fever,hotpotqa,triviaqa,wow,structured_zeroshot-0-True-Truejs=0.01#expand=1#n_tokens=100#for_beir'
    # model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-5-True-Truejs=0.01#expand=5#n_tokens=100#no_reg'
    model_file = 'splade-fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow-4-True-Truejs=0.01#expand=5#n_tokens=100#remove_different_task_diffuse'
    

    for dataset in kilt_dataset:
        mpt_mode = 'False'
        use_prompt = 'True'
        use_v2 = 'True'
        n_tokens = '100'
        share_expand = '5'
        commend = f"../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False dataset={dataset} local_process=False \
        num_workers=1 use_prompt={use_prompt} mpt_mode={mpt_mode} n_gpu=8 use_v2={use_v2}\
        n_tokens={n_tokens} share_expand={share_expand} do_train=False load_question_name=\\\'{model_file}\\\' \
        customer_keys=remove_diffuse"
        # n_tokens=50 share_expand=10 do_train=False load_question_name=\\\'{model_file}\\\'"
        f.write(commend + '\n')
