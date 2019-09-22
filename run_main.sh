#!/usr/bin/env bash
release_path=tvqa_plus_stage_features

debug_vcpt_path=${release_path}/bottom_up_visual_sen_hq_bbt_100_debug.pickle
vcpt_path=${release_path}/tvqa_bbt_frcn_vg_hq_20_100.json
vfeat_path=${release_path}/tvqa_bbt_frcn_vg_hq_20_100_pca.h5

train_path=${release_path}/tvqa_plus_train_preprocessed.json
valid_path=${release_path}/tvqa_plus_valid_preprocessed.json
qa_bert_path=${release_path}/bbt_qa_s_tokenized_bert_sub_qa_tuned_new_qid.h5
sub_bert_path=${release_path}/bbt_sub_s_tokenized_bert_sub_qa_tuned.h5
sub_path=${release_path}/tvqa_plus_subtitles.json

word2idx_path=${release_path}/word2idx.json
frm_cnt_path=${release_path}/frm_cnt_cache.json

if [[ $1 == 'debug' ]]; then
    echo "debug mode"
    python main.py \
    --train_path ${train_path} \
    --valid_path ${valid_path} \
    --sub_path ${sub_path} \
    --qa_bert_path ${qa_bert_path} \
    --sub_bert_path ${sub_bert_path} \
    --vcpt_path ${debug_vcpt_path} \
    --vfeat_path ${vfeat_path} \
    --word2idx_path ${word2idx_path} \
    --frm_cnt_path ${frm_cnt_path} \
    --use_sup_att \
    --debug \
    ${@:2}
else  # start with the 1st argument
    python main.py \
    --train_path ${train_path} \
    --valid_path ${valid_path} \
    --sub_path ${sub_path} \
    --qa_bert_path ${qa_bert_path} \
    --sub_bert_path ${sub_bert_path} \
    --vcpt_path ${vcpt_path} \
    --vfeat_path ${vfeat_path} \
    --word2idx_path ${word2idx_path} \
    --frm_cnt_path ${frm_cnt_path} \
    --use_sup_att \
    ${@:1}
fi
