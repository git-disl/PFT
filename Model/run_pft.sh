python gen_seq.py
python gen_pretrain_data.py --dupe_factor=10 --do_eval=False
python gen_pseudo_pair.py

python run_pretrain.py --checkpointDir=lm_ckpt_dir
python run_pseudo_finetun.py --init_checkpoint=lm_ckpt_dir/model_104000 --checkpointDir=pft_ckpt_dir

# output embedding
python run_embed.py --init_checkpoint=pft_ckpt_dir/model_6000
