../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.001 share_expand=5 customer_keys=\'js=0.001#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.005 share_expand=5 customer_keys=\'js=0.005#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.01 share_expand=5 customer_keys=\'js=0.01#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.02 share_expand=5 customer_keys=\'js=0.02#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.05 share_expand=5 customer_keys=\'js=0.05#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.1 share_expand=5 customer_keys=\'js=0.1#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.2 share_expand=5 customer_keys=\'js=0.2#expand=5#n_tokens=100\' do_val_on=nq
../../ks.sh tune ; python prompt_tune_retriever.py dense_model=False learning_rate=2e-5 dataset=\'fever,aidayago2,nq,hotpotqa,triviaqa,structured_zeroshot,wow\' freeze_d_model=True ng_count=19 train_batch=24 eval_batch=64 local_process=False num_workers=1 warmup_steps=5000 num_train_epochs=5 n_gpu=8 use_prompt=True n_tokens=100 module_learning_rate=1e-3 use_v2=True js_weight=0.5 share_expand=5 customer_keys=\'js=0.5#expand=5#n_tokens=100\' do_val_on=nq