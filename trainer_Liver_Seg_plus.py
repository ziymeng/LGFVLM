import os
import shutil
import time
import ast

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather
from utils.utils import dice, resample_3d, TAO_ORGAN_NAME
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def train_epoch(model, t1_loader, t1c_loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    t1_align_loss = 0.0
    t1c_align_loss = 0.0
    save_log_dir = args.logdir
    for idx, (batch_t1, batch_t1c) in enumerate(zip(t1_loader, t1c_loader)):
        if isinstance(batch_t1, list) and isinstance(batch_t1, list):
            t1_data, t1_mask = batch_t1
            t1c_data, t1c_mask = batch_t1c
        else:
            t1_data, t1_mask = batch_t1["img_T2WI"], batch_t1["mask_T2WI"]
            t1c_data, t1c_mask = batch_t1c["img_T1IP"], batch_t1c["mask_T1IP"]
        t1_data, t1_mask, t1c_data, t1c_mask = t1_data.cuda(args.rank), t1_mask.cuda(args.rank), t1c_data.cuda(args.rank), t1c_mask.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            if epoch < args.start_fusion_epoch:
                fuse_out, t1_seg, t1c_seg = model(t1_data, t1c_data)
                t1_seg_loss = loss_func(t1_seg, t1_mask)
                t1_fuse_seg_loss = loss_func(fuse_out, t1_mask)
                t1_seg_loss_all = 0.4* t1_seg_loss + 0.6*t1_fuse_seg_loss
                t1c_seg_loss = loss_func(t1c_seg, t1c_mask)
                t1c_fuse_seg_loss = loss_func(fuse_out, t1c_mask)
                t1c_seg_loss_all = 0.4* t1c_seg_loss + 0.6*t1c_fuse_seg_loss
                loss = (t1_seg_loss_all + t1c_seg_loss_all) / 2
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < t1_loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "t1_seg_loss: {:.4f}".format(t1_seg_loss),
                "t1c_seg_loss: {:.4f}".format(t1c_seg_loss),
                "time {:.2f}s".format(time.time() - start_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                    "loss: {:.4f}".format(run_loss.avg),
                    "t1_seg_loss: {:.4f}".format(t1_seg_loss),
                    "t1c_seg_loss: {:.4f}".format(t1c_seg_loss),
                    "time {:.2f}s".format(time.time() - start_time),file=f
                )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg


def val_epoch(model, t1_loader, t1c_loader, epoch, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    all_avg_dice =[]
    start_time = time.time()
    model_inferer =None
    save_log_dir = args.logdir
    nun_class = args.out_channels
    with torch.no_grad():
        for idx, (batch_t1, batch_t1c) in enumerate(zip(t1_loader,t1c_loader)):
            if isinstance(batch_t1, list) and isinstance(batch_t1c, list):
                t1_data, t1_mask = batch_t1
                t1c_data, t1c_mask = batch_t1c
            else:
                t1_data, t1_mask = batch_t1["img_T2WI"], batch_t1["mask_T2WI"]
                t1c_data, t1c_mask = batch_t1c["img_T1IP"], batch_t1c["mask_T1IP"]
            t1_data, t1_mask, t1c_data, t1c_mask = t1_data.cuda(args.rank), t1_mask.cuda(
                args.rank), t1c_data.cuda(args.rank), t1c_mask.cuda(args.rank)
            _, _, h, w, d = t1_mask.shape
            target_shape = (h, w, d)
            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    fuse_out, t1_seg_out, t1c_seg_out = model_inferer(t1_data, t1c_data)
                else:
                    fuse_out, t1_seg_out, t1c_seg_out= model(t1_data, t1c_data)
            if not t1_seg_out.is_cuda:
                t1_mask, t1c_mask = t1_mask.cpu(), t1c_mask.cpu()
            val_outputs = torch.softmax(fuse_out, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = t1_mask.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            organ_dice = []
            for i in range(1, nun_class):
                spleen_dice = dice(val_outputs == i, val_labels == i)
                organ_dice.append(spleen_dice)
            avg_dice = np.mean(organ_dice)
            print("avg_dice:{}".format(avg_dice))
            all_avg_dice.append(avg_dice)
            if args.rank == 0:
                avg_acc = avg_dice
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
                with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                    print(
                        "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(t1_loader)),
                        "acc",
                        avg_acc,
                        "time {:.2f}s".format(time.time() - start_time),file=f
                    )
            start_time = time.time()
    return np.mean(all_avg_dice)

def save_checkpoint(model, epoch, args, filename="model_v1.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

def run_training(
    model,
    t1_train_loader,
    t1_val_loader,
    t1c_train_loader,
    t1c_val_loader,
    optimizer,
    loss_func,
    args,
    scheduler=None,
    start_epoch=0,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    save_log_dir = args.logdir
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            t1_train_loader.sampler.set_epoch(epoch)
            t1c_train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, t1_train_loader,t1c_train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                print(
                    "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                    "loss: {:.4f}".format(train_loss),
                    "time {:.2f}s".format(time.time() - epoch_time),file=f
                )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0 or (epoch + 1)== args.max_epochs:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                t1_val_loader,
                t1c_val_loader,
                epoch=epoch,
                args=args,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                    print(
                        "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                        "acc",
                        val_avg_acc,
                        "time {:.2f}s".format(time.time() - epoch_time),file=f
                    )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc),file=f)
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final_v1.pt")
                if b_new_best:
                    print("Copying to model_v1.pt new best model!!!!")
                    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
                        print("Copying to model_v1.pt new best model!!!!",file=f)
                    shutil.copyfile(os.path.join(args.logdir, "model_final_v1.pt"), os.path.join(args.logdir, "model_v1.pt"))
        if scheduler is not None:
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    with open(os.path.join(save_log_dir, 'log.txt'), 'a') as f:
        print("Training Finished !, Best Accuracy: ", val_acc_max,file=f)

    return val_acc_max
