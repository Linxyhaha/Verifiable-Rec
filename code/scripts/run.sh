export TOKENIZERS_PARALLELISM=false

category=${1} # dataset

nn=8 # number of GPUs
model_class="LatentModelwithVerifier_RATT_MS_MoV" # VRec
n_verifier=2 # number of verifiers
n_thought=1 # number of reasoning steps

tdir=test
train_file=$(ls -f ../data/${category}/train/${category}.csv)
eval_file=$(ls -f ../data/${category}/valid/${category}.csv)
test_file=$(ls -f ../data/${category}/${tdir}/${category}.csv)
info_file=$(ls -f ../data/${category}/info/${category}.txt)

echo ${train_file} ${test_file} ${info_file} ${eval_file} ${asin2catid_file}
port=$((RANDOM % 5000 + 10000))

for ((i=0; i<nn; i++)); do
    array+=($i)
done

n_cluster=25 # number of classes for group-level preference prediction
asin2catid_file=$(ls -f ../data/${category}/info/${category}_asin2cat_qwen_cluster_${n_cluster}.npy)
asin2clusterid_file=$(ls -f ../data/${category}/info/${category}_asin2cat_cf_cluster_${n_cluster}.npy)
correct_dict_file_train=$(ls -f ../data/${category}/${n_thought}T/correct_dict_training.npy) 
correct_dict_file_valid=$(ls -f ../data/${category}/${n_thought}T/correct_dict_validation.npy) 
        
suffix=-${n_cluster}cluster

# training stage 1: freeze reasoning LLM, with only verifier training
BASE_MODEL=./output_dir/${category}/pretrain_${n_thought}T # TODO
task_name="verifier_pretrain/${category}_${n_verifier}v_${n_thought}T"
out_path="./output_dir/${category}/${n_verifier}v_${n_thought}T${suffix}"
mkdir -p ${out_path}

stage2_lr=1e-4
cuda_ids=$(seq -s, 0 $((nn - 1)))
export CUDA_VISIBLE_DEVICES=${cuda_ids}
task_name="${category}_${n_verifier}v_${n_thought}T"
out=${out_path}/${stage2_lr}
mkdir -p ${out}

# verifier training
torchrun --nnodes=1 --nproc_per_node=${nn} --master_port=$port \
    src/pretrain_verifier/train.py \
    --base_model $BASE_MODEL \
    --model_class ${model_class} \
    --train_file ${train_file} --eval_file ${eval_file} --asin2catid_file ${asin2catid_file} \
    --asin2clusterid_file ${asin2clusterid_file} \
    --output_dir ${out} --category ${category} --sample ${n_sample} \
    --correct_dict_file_train ${correct_dict_file_train} \
    --correct_dict_file_valid ${correct_dict_file_valid} \
    --n_thought ${n_thought} \
    --num_verifier ${n_verifier} \
    --learning_rate ${stage2_lr} \
    --num_epochs 20 \
    --micro_batch_size 16 \
    --batch_size 128 \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

# training stage 2: train both LLM and verifier with multi thoughts
beta=1 # beta -> strength of verifier loss
gamma=1 # gamma -> strength of monotonicity regularization
BASE_MODEL=${out_path}/${stage2_lr}lr 
learning_rate=5e-5

suffix_base=VRec_${n_thought}T_${topk}_${n_cluster}cluster_${stage2_lr}lr_s3_${learning_rate}lr_${beta}beta_${gamma}gamma 
echo "run suffix: ${suffix_base}"

suffix=${suffix_base}
task_name="${category}" 
out_path=./output_dir/${category}/${n_verifier}v_${n_thought}T/${suffix}

cuda_ids=$(seq -s, 0 $((nn - 1)))
export CUDA_VISIBLE_DEVICES=$cuda_ids

out=${out_path}

# fine-tuning
torchrun --nnodes=1 --nproc_per_node=${nn} --master_port=$port \
    src/finetune/train.py \
    --base_model $BASE_MODEL \
    --model_class ${model_class} \
    --train_file ${train_file} --eval_file ${eval_file} --asin2catid_file ${asin2catid_file} \
    --asin2clusterid_file ${asin2clusterid_file} \
    --output_dir ${out} --category ${category} --sample ${n_sample} \
    --n_thought ${n_thought} \
    --num_verifier ${n_verifier} \
    --beta ${beta} \
    --gamma ${gamma} \
    --learning_rate ${learning_rate} \
    --micro_batch_size 8 \
    --batch_size 128 \
    --num_epochs ${num_epochs} \
    > >(tee -a ${out}/${task_name}.log) 2> >(tee -a ${out}/${task_name}.err >&2)

mkdir -p ./output_dir/${category}_finetune${suffix}
python src/utils/split.py --input_path $test_file --output_path ./output_dir/${category}_finetune${suffix} --nn ${nn}

model_ckpt=${out}
for (( i=0; i<$nn; i++ )); do
    export CUDA_VISIBLE_DEVICES=${array[$i]}
    mkdir -p ./output_dir/${category}_finetune${suffix}
    echo ${category}_finetune/${i}.csv
    python src/finetune/eval.py --batch_size 24 \
            --base_model $model_ckpt --sample ${n_sample} \
            --model_class ${model_class} \
            --info_file ${info_file} --category ${category} \
            --test_data_path ./output_dir/${category}_finetune${suffix}/${i}.csv \
            --asin2catid_file ${asin2catid_file} \
            --asin2clusterid_file ${asin2clusterid_file} \
            --result_json_data ./output_dir/${category}_finetune${suffix}/${i}.json &
done
wait

python src/utils/merge.py --input_path ./output_dir/${category}_finetune${suffix}/ \
        --output_path ${model_ckpt}/final_result.json --nn ${nn}
python src/utils/calc.py --path ${model_ckpt}/final_result.json \
        --item_path ${info_file} \
        > >(tee -a ${out}/${task_name}_test.log) 2> >(tee -a ${out}/${task_name}.err >&2)