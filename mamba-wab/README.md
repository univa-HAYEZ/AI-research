python mamba-wab/wab_eval_mamba_libero_v2.py \
--csv mamba-wab/libero_instructions_compressed.csv \
--model mistralai/Mamba-Codestral-7B-v0.1 \
--device cuda \
--history_k 3 \
--max_steps 8 \
--out_json mamba-wab/summary.json \
--out_per_episode mamba-wab/per_episode.csv
								


