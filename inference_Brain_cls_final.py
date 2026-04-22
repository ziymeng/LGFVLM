import os
import time 
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import json
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, classification_report

from model.CLS.mm_classification_Foundation_model_plus import Foundation_Model_Classification
# from model.CLS.mm_classification_SwinUnter import MM_SwinUnter_Classification
from monai.utils import set_determinism

# 这行代码内部会自动处理 numpy, torch, random 以及 dataloader 的 worker
set_determinism(seed=42)
import argparse
from utils.MM_CLS_Brain_data_utils import get_loader
import ast
from torch.cuda.amp import GradScaler, autocast
from utils.metrics import *
from collections import OrderedDict
from torchcam.methods import SmoothGradCAMpp

parser = argparse.ArgumentParser(description="VLM cls pipeline")
parser.add_argument('--model_name', default="", type=str)
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--data_dir", default="./dataset/mm_brain_data/", type=str, help="dataset directory")
parser.add_argument("--json_list", default="mm_brain_cls_seg_fold5.json", type=str, help="dataset json file")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument('--pretrain_dir', default=f"./pretrained_models/Foundation_model.pth", type=str)
parser.add_argument('--CLIP_text_pretrain_dir', default=f"./Text-emmbedding-gen/CLIP_Brain_txt_encoding.pth", type=str)
parser.add_argument('--Bert_text_pretrain_dir', default=f"./Text-emmbedding-gen/Brain_bert_txt_encoding.pth", type=str)
parser.add_argument("--logdir", default="VLM_brain_cls_2", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--gpu", default=0, type=int, help="number of gpu")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=1, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument('--text_encoding_type', default='word_embedding', choices=['word_embedding','rand_embedding', 'None'],
                    help='the type of encoding: rand_embedding or word_embedding')
parser.add_argument('--text_prompt_name', default='Bert_embedding', choices=['CLIP_embedding','Bert_embedding', 'None'], help='text embedding type')
parser.add_argument('--use_text_prompt', default=1, type=int)
parser.add_argument('--text_prompt_loss', default=1, type=int)
parser.add_argument('--fusion_module', default='Cross_Attention', choices=['DoubleAttention', 'Attention_Fusion', 'Cross_Attention', 'SingleAttention', 'Concat_Fusion'])
parser.add_argument('--res_depth', default=50, type=int, choices=[18, 34, 50, 101, 152])
parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")

def main():
    args = parser.parse_args()
    set_determinism(seed=42)

    args.test_mode = True
    args.amp = not args.noamp
    ### load dataset
    crop_organ_loader = get_loader(args,train_modality='crop')
    Brain_loader = get_loader(args,train_modality='nocrop')
    
    model = Foundation_Model_Classification(n_class=args.out_channels,
                                            text_prompt=args.use_text_prompt,
                                            fusion_module=args.fusion_module,
                                            text_prompt_name=args.text_prompt_name,
                                            text_encoding=args.text_encoding_type,
                                            res_depth=args.res_depth)

    model.load_params(torch.load(args.pretrain_dir, map_location='cpu')['net'])  # load pretrain model
    args.use_text_prompt = True
    word_embedding = torch.load(args.CLIP_text_pretrain_dir)
    model.VLM_branch.organ_embedding.data = word_embedding.float()
    model.load_state_dict(torch.load(
        "checkpoints/Brain_data/cls/model.pt"), strict=False)
    model.cuda(args.gpu)

    if args.text_prompt_name=="CLIP_embedding":
        args.use_text_prompt=True
        word_embedding = torch.load(args.CLIP_text_pretrain_dir)
        model.VLM_branch.organ_embedding.data = word_embedding.float()
        print('load CLIP word embedding')
    elif args.text_prompt_name=="Bert_embedding":
        args.use_text_prompt=True
        word_embedding = torch.load(args.Bert_text_pretrain_dir)
        model.VLM_branch.organ_embedding.data = word_embedding.float()
        print('load Bert word embedding')

    inference_cls(model, crop_organ_loader, Brain_loader, args)

def inference_cls(model, crop_organ_loader, Liver_loader, args):
    model.eval()
    # cam_extractor = SmoothGradCAMpp(model,target_layer=model.blocks2[-1].mlp,input_shape=(8,14,112,112))
    y_pred_prob = []
    eval_time = time.time()
    start_time = time.time()
    all_acc = []
    all_precision = []
    all_F1_score = []
    ind = 0
    with torch.no_grad():
        for idx, (batch_crop, batch_brain) in enumerate(zip(crop_organ_loader, Liver_loader)):
            crop_data, crop_target, label = batch_crop["img_FLAIR"], batch_crop["mask_seg"], batch_crop["label"]
            unique_label = list({item.split('/')[-1] for item in label})
            parsed_labels = [ast.literal_eval(label) for label in unique_label]
            labels = np.array(parsed_labels, dtype=float)
            labels = torch.FloatTensor(labels)
            brain_data, brain_target = batch_brain["img_FLAIR"], batch_brain["mask_seg"]
            crop_data, crop_target, brain_data, brain_target, labels = crop_data.cuda(args.rank), crop_target.cuda(
            args.rank), brain_data.cuda(args.rank), brain_target.cuda(args.rank), labels.cuda(args.rank)
            _, _, h, w, d = crop_target.shape

            with autocast(enabled=args.amp):
                val_preds, _ = model(crop_data, brain_data) 

            img_name = batch_crop["img_FLAIR_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            y_pred_prob.append(val_preds.cpu().numpy())
            val_preds[val_preds >= 0] = 1
            val_preds[val_preds < 0] = 0

            val_labels = labels.cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            # val_preds = np.where(labels == 1, val_preds, 0)

            case_acc, case_prec, case_f1_score = cls_score(val_preds, val_labels, y_pred_prob)
            all_acc.append(case_acc)
            all_precision.append(case_prec)
            all_F1_score.append(case_f1_score)
            if case_acc>-1:
                ind = ind + 1

    avg_acc = np.mean(all_acc)
    avg_precision = np.mean(all_precision)
    avg_F1_score = np.mean(all_F1_score)
    print('Brain dataset classification inference:\n', 'avg_acc:', avg_acc, "avg_precision", avg_precision, "avg_F1_score:",avg_F1_score, 'time {:.2f}s'.format(time.time() - eval_time))

def cls_score(pred, label, prob):
    """
    pred:  [[1, 0, 0, 0], [0, 0, 0, 1]]
    label: [[0, 0, 1, 1], [1, 1, 1, 1]]
    """

    # 1. 准备数据
    # pred = np.where(label == 1, pred, 0)
    labels = torch.tensor(label)
    preds = torch.tensor(pred)

    # 2. 转换为 Numpy 并展平 (Flatten)
    y_true = labels.view(-1).detach().cpu().numpy()
    y_pred = preds.view(-1).detach().cpu().numpy()

    # 3. 计算指标
    acc = accuracy_score(y_true, y_pred)
    # zero_division=0 防止分母为0时报错
    prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro',zero_division=0)


    return acc, prec, f1

if __name__ == "__main__":
    main()


