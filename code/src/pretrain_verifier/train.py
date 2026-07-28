import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
import fire
import torch
import numpy as np
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
)

from layers import *
from reasoning_dataset import *
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from finetune.layers import *
from finetune.reasoning_dataset import *
# torch.backends.cuda.enable_cudnn_sdp(False)
import ipdb

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    # model/data params
    base_model: str = "/your/path/to/qwen/model/",  # the only required argument
    train_file: str="/train/file/path/",
    eval_file: str="/valid/file/path",
    asin2catid_file="/cluster/path",
    asin2clusterid_file="/cluster/path",
    output_dir: str = "./output_test/",
    sample: int = 1024,
    seed: int = 0,
    # attention hyper parameters
    end_k: int = -1,
    # reasoning step
    n_thought: int = 1,
    # reasoning model
    model_class: str = "LatentModelwithVerifier_RATT",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    # ddp
    local_rank: int = 0,
    deepspeed: str ="./deepspeed.json",
    # others
    category: str="CDs_and_Vinyl",
    K: int = 0,
    # verifier
    num_verifier: int = 1,
    v_weight: float = 1,
    mono_weight: float = 0,
):
    set_seed(seed)
    category_dict = {"MicroLens":"micro-videos", "Office_Products": "office products", "Books": "books", "steam": "games", "CDs_and_Vinyl": "musics", "Toys_and_Games": "toys and games", "Video_Games": "video games", "Musical_Instruments": "music instruments", "Sports_and_Outdoors": "sports and outdoors", "Pet_Supplies": "pet supplies", "Arts_Crafts_and_Sewing": "arts products", "Movies": "movie", "yelp": "resturant"}
    # print(category)
    category = category_dict[category]

    asincat = np.load(asin2catid_file, allow_pickle=True).item()
    num_category = len(set(list(asincat.values())))
    asincluster = np.load(asin2clusterid_file, allow_pickle=True).item()
    num_cluster = len(set(list(asincluster.values())))

    Model = eval(model_class)

    assert (
        base_model
    )
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    config = AutoConfig.from_pretrained(base_model)

    config.n_thought = n_thought
    config.num_verifier = num_verifier
    config.num_category = num_category
    config.verifier_dims = [num_category, num_cluster]
    config.v_weight = v_weight
    config.mono_weight = mono_weight

    model = Model.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    # model.attention.end_k = end_k
    model.config.loss_type = "ce"
    

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    additional_special_tokens = tokenizer.additional_special_tokens
    additional_special_tokens.append("<|Thought|>")
    num_added_toks = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
    model.resize_token_embeddings(len(tokenizer))
    assert tokenizer("<|Thought|>")['input_ids'][0] == len(tokenizer) - 1

    # stage 2: freeze the reasoning LLM (+RATT), train only the verifier
    model.freeze_llm_params()

    ##################
    ##############################################################################
    if model_class == "LatentModelwithVerifier":
        Customize_Dataset = RecDataset  
    else:
        Customize_Dataset = LatentRDataset_Pretrain

    train_data = Customize_Dataset(train_file=train_file, asin2catid_file=asin2catid_file, asin2clusterid_file=asin2clusterid_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, seed=seed, category=category, K = K)
    val_data = Customize_Dataset(train_file=eval_file, asin2catid_file=asin2catid_file, asin2clusterid_file=asin2clusterid_file, tokenizer=tokenizer, max_len=cutoff_len,  sample=sample, category=category, K = K)

    if int(os.environ.get("LOCAL_RANK")) == 0:
        print("*** LOAD DATA FINISHED")

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    from datasets import Dataset as HFDataset
    # print("train_data 0 keys:", train_data[0].keys())
    hf_train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
    hf_val_dataset = HFDataset.from_dict({k: [v[k] for v in val_data] for k in val_data[0].keys()})
    trainer = transformers.Trainer(
        # deepspeed=deepspeed,
        model=model,
        train_dataset=hf_train_dataset,
        eval_dataset=hf_val_dataset,
        processing_class=tokenizer,
        args=transformers.TrainingArguments(
            # deepspeed=deepspeed,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            # ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=True,
            group_by_length=group_by_length,
            report_to="none",
            remove_unused_columns=False
        ),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks = [EarlyStoppingCallback(early_stopping_patience=1)],
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
