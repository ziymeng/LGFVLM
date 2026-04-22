from typing import Optional
import transformers
from transformers import Trainer
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import ITRDataset
# from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from LaMed.src.model.Swin_CLIP import M3DCLIP, M3DCLIPConfig
from transformers import BertTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file
import os
# import transformers.modeling_utils import unwrap_model
from numpy import inf
from accelerate import Accelerator, DistributedType
from torch.distributed import is_initialized, get_rank
from torch.utils.data import DataLoader, Dataset
import json
# import torch.nn.utils.convert_parameters as convert_parameters

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="M3D/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
    local_loss: bool = field(default=False)

    pretrained_model: str = field(default=None)
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(128, 128, 128))
    # patch_size: tuple = field(default=(16, 16, 4))

    hidden_size: int = field(default=768)
    # mlp_dim: int = field(default=3072)
    # num_layers: int = field(default=12)
    # num_heads: int = field(default=12)
    # pos_embed: str = field(default="perceptron")
    # dropout_rate: float = field(default=0.0)
    # spatial_dims: int = field(default=3)
    max_text_len: int = field(default=128)
    vocab_size: int = field(default=30522)
    
@dataclass
class DataArguments:
    data_root: str = field(default="./mm_pretrain_data/LLD_MMRI_data/crop_all_npz_data", metadata={"help": "Root directory for all data."})
    # caption data
    cap_data_path: str = field(default="./mm_pretrain_data/LLD_MMRI_data/liver_pretrain_fine-grained.json", metadata={"help": "Path to caption data."})
    max_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)

    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = True

    # config in bash file
    bf16: bool = True
    output_dir: str = "./LaMed/output/BioBert_zxg"
    num_train_epochs: int = 100
    per_device_train_batch_size: int = 2 #32
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    eval_steps: float = 0.04 # 0.04
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 1
    learning_rate: float = 1e-4 #1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001 # 0.001
    gradient_checkpointing: bool = False # train fast
    dataloader_pin_memory: bool = True # fast
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    preds = torch.argmax(logits, dim=-1)
    return preds

@dataclass
class DataCollator:
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        images, texts, input_ids, attention_mask = tuple(
            [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask'))

        images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
        input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
        attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size

        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        return_dict = dict(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,

        )

        return return_dict

# def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str, bias="none"):
#     if trainer.args.hf_deepspeed_config.config['zero_optimization']['stage'] == 3:
#         state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
#     else:
#         state_dict = trainer.accelerator.get_state_dict(trainer.model)
#     if trainer.args.should_save and trainer.args.local_rank == 0:
#         trainer._save(output_dir, state_dict = state_dict)
        


def main():
    print("zxg: step1")
    # torch.distributed.init_process_group('nccl', init_method='file:///home/.../my_file', world_size=1, rank=0)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    accelerator = Accelerator()
    #print("zxg: model_args",model_args)
    #print("zxg: data_args",data_args)
    #print("zxg: training_args",training_args)

    # tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path)
    #./LaMed/pretrained_model/bert_base_uncased
    config = M3DCLIPConfig.from_dict(vars(model_args))
    model = M3DCLIP(config)
    # pretrain = "./pretrained_model/supervised_suprem_swinunetr_2100.pth"
    pretrain = "./pretrained_model/Foundation_model.pth"
    model.load_params(torch.load(pretrain, map_location='cpu')['net'])
    

    if model_args.pretrained_model:  # default is None
        ckpt = torch.load(model_args.pretrained_model)
        model.load_state_dict(ckpt, strict=True)
        print("load pretrained model.")
    print("zxg:!!!")
    with open('name_category_CT_CLIP_1k.json', 'r') as f:
        name_to_category = json.load(f)
    
    train_dataset = ITRDataset(data_args, tokenizer, name_to_category, mode='train')
    eval_dataset = ITRDataset(data_args, tokenizer, name_to_category, mode='validation')

    if model_args.gather_loss and not model_args.local_loss:
        gather_all = True
    else:
        gather_all = False
    data_collator = DataCollator(gather_all)

    # train_loader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
    # eval_loader = DataLoader(eval_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=True)
    # model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)  # GPUs share model
    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        compute_metrics=compute_metrics,
                        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                      )
    
    trainer.accelerator = accelerator
    '''
    print("zxg:trainer.accelerator.state",trainer.accelerator.state)
    print("zxg:trainer.is_deepspeed_enabled",trainer.is_deepspeed_enabled)
    print("zxg:trainer.args.deepspeed",trainer.args.deepspeed)
    print("zxg:trainer.args.deepspeed_plugin",trainer.args.deepspeed_plugin)
    '''

    #print("zxg:trainer.args.hf_deepspeed_config",trainer.args.hf_deepspeed_config)
    # exit()
    # if you want to resume your training, pls set the checkpoint in trainer.train(resume_from_checkpoint="")
    trainer.train()
    # print("Before save_state")
    # trainer.save_state()
    # print("After save_state")
        # trainer.accelerator.wait_for_everyone()
        # unwrapped_model = trainer.accelerator.unwrap_model(model)
        
        
        # unwrapped_model.save_pretrained(
        #     training_args.output_dir,
        #     is_main_process=trainer.accelerator.is_main_process,
        #     save_function = trainer.accelerator.save,
        #     state_dict = trainer.accelerator.get_state_dict(model),
        # )
        # if trainer.accelerator.is_main_process:
        #     tokenizer.save_pretrained(training_args.output_dir)
    
    # model = unwrap_model(model)
    # if is_initialized() and get_rank()==0:

    #trainer.save_state()
    # for param in model.parameters(): 
        # param.data = param.data.contiguous()



    model.config.save_pretrained(os.path.join(training_args.output_dir,'model_config'))
    # print("after config.save_pretrain")
    model.save_pretrained(os.path.join(training_args.output_dir, 'model_pretrained'))
    # print("after model.save_pretrain")
    tokenizer.save_pretrained(os.path.join(training_args.output_dir, 'tokenizer_pretrained'))
    # print("after tokenizer.save_pretrain")
    state_dict = model.state_dict()
    # print("after model.state_dict()")
    # torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))
    torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.pt'))
    # print(trainer.accelerator)
    # print(trainer.accelerator.distributed_type)
    
    # model.save_checkpoint(os.path.join(training_args.output_dir, 'ckpt.bin'))
    # model = trainer.accelerator.unwrap_model(model)
    # new_model = trainer.accelerator.get_state_dict(model)
    # torch.save(new_model.state_dict(), os.path.join(training_args.output_dir, 'new_model.bin'))
    
    # torch.save(new_state_dict, os.path.join(training_args.output_dir, 'model_params_accelerator_state_dict.bin'))
    # prin('after save new sd')
    # model.save_pretrained(os.path.join(training_args.output_dir,'accelerator_state_dict'), state_dict = new_state_dict, safe_serialization=False)
    # # trainer.save_model(training_args.output_dir)
    # print("after save pretrained new sd")

if __name__ == "__main__":
    main()
