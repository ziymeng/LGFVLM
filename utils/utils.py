# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import torch
from monai.inferers import sliding_window_inference

'''
AMOS dataset label:
"labels": {
    "00": "background",
    "01": "Spleen",
    "02": "Right Kidney",
    "03": "Left Kidney",
    "04": "Gall Bladder",
    "05": "Esophagus",
    "06": "Liver",
    "07": "Stomach",
    "08": "Aorta",
    "09": "Inferior Vena Cava",
    "10": "Pancreas",
    "11": "Right Adrenal Gland",
    "12": "Left Adrenal Gland",
    "13": "Duodenum",
    "14": "Bladder",
    "15": "Prostate"
}
'''

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
                'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Pancreas',
                'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum']

TAO_ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus',
                'Liver', 'Stomach', 'Aorta', 'Inferior Vena Cava', 'Pancreas',
                'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum']

def resample_3d(img, target_size):
    if len(img.shape)>3:
        _, _, imx, imy, imz = img.shape
        tx, ty, tz = target_size
    else:
        imx, imy, imz = img.shape
        tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def compute_dice(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.
    Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    if isinstance(mask_gt, np.ndarray):
        mask_gt = mask_gt.astype(bool)
        mask_pred = mask_pred.astype(bool)
    elif isinstance(mask_gt, torch.Tensor):
        mask_gt = mask_gt.bool()
        mask_pred = mask_pred.bool()
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum

def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.tensor(preds)  # 转换为 Tensor
    labels = torch.tensor(labels)  # 转换为 Tensor
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)


    # print(dice, recall, precision)
    if spe_sen:
        # return dice, recall, precision, specificity
        return dice
    else:
        # return dice, recall, precision
        return dice

def segment_with_fusion(model, input_tensor,device,args):
    ####多种增强方式输入
    """
        使用 8 种数据增强方式进行分割，并融合结果
        """
    #  计算不同增强方式的预测
    outputs = []
    # 原始输入
    output = sliding_window_inference(
        input_tensor, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
    )
    output = torch.softmax(output, 1).cpu().numpy()
    output = np.argmax(output, axis=1).astype(np.uint8)[0]
    outputs.append(output)
    # 旋转
    for angle in [45, 90, 180, 270]:
        rotated_input = rotate_tensor_3d(input_tensor, angle)
        output = sliding_window_inference(
            rotated_input, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
        )
        output = inverse_rotate_tensor_3d(output, angle)
        output = torch.softmax(output, 1).cpu().numpy()
        output = np.argmax(output, axis=1).astype(np.uint8)[0]
        outputs.append(output)

    # 翻转
    for axis in [2, 3]:  # 水平、垂直、深度翻转
        flipped_input = flip_tensor_3d(input_tensor, axis)
        output = sliding_window_inference(
            flipped_input, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
        )
        output = flip_tensor_3d(output, axis)  # 反向翻转恢复
        output = torch.softmax(output, 1).cpu().numpy()
        output = np.argmax(output, axis=1).astype(np.uint8)[0]
        outputs.append(output)
        # 采用融合策略（最大置信度融合）
    outputs = [torch.from_numpy(out).to(device) if isinstance(out, np.ndarray) else out.to(device) for out in
               outputs]
    fused_output = torch.maximum(outputs[0], outputs[1])
    for i in range(2, len(outputs)):
        fused_output = torch.maximum(fused_output, outputs[i])

    fused_output = fused_output.cpu().numpy()

    return fused_output

def rotate_tensor_3d(tensor, angle):
    """旋转 3D Tensor（Depth, Height, Width），保持通道顺序"""
    if angle == 90:
        return tensor.rot90(1, [2, 3])  # 沿 (H, W) 旋转 90°
    elif angle == 180:
        return tensor.rot90(2, [2, 3])  # 旋转 180°
    elif angle == 270:
        return tensor.rot90(3, [2, 3])  # 旋转 270°
    else:
        return tensor  # 原始方向

def inverse_rotate_tensor_3d(tensor, angle):
    """反向旋转，将预测结果恢复到原始方向"""
    if angle == 90:
        return tensor.rot90(3, [2, 3])  # 旋转 -90°
    elif angle == 180:
        return tensor.rot90(2, [2, 3])  # 旋转 -180°
    elif angle == 270:
        return tensor.rot90(1, [2, 3])  # 旋转 -270°
    else:
        return tensor  # 原始方向

def flip_tensor_3d(tensor, axis):
    """翻转 3D Tensor，axis=2（水平），axis=3（垂直），axis=1（深度）"""
    return torch.flip(tensor, dims=[axis])

def get_dice_score(prev_masks, gt3D): #refer to SAM-Med3D
    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    pred_masks = (prev_masks >= 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))

    # 检查是否有有效的样本
    if len(dice_list) == 0:
        print("Warning: dice_list is empty. Returning default value 0.0")
        result = 0.0
    else:
        # result = (sum(dice_list) / len(dice_list)).item()
        result = (sum(dice_list) / len(dice_list))
    return result

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def extract_surface(mask: np.ndarray) -> np.ndarray:
    """
    提取 3D 二值掩膜的表面体素 (surface voxels)。
    返回形状与 mask 相同的 bool 数组，仅表面为 True。
    """
    mask = mask.astype(bool)
    # 使用最大值滤波查看局部 3x3x3(neighborhood) 是否全为 True
    # 非全为 True 且自身为 True 的位置视为表面
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)))
    surface = mask & ~eroded
    return surface

def assd_3d(pred, gt, spacing, threshold):
    """
    计算 3D 平均对称表面距离 ASSD (Average Symmetric Surface Distance)

    参数
    ----
    pred : np.ndarray
        预测掩膜，形状可为 [D, H, W] 或带 batch/channel 但最终会 squeeze。
    gt : np.ndarray
        GT 掩膜，形状可为 [D, H, W] 或 [1,1,D,H,W] 等，最终会 squeeze。
    spacing : tuple(float, float, float)
        体素间距 (z, y, x)。默认等距 (1,1,1)。
    threshold : float | None
        若不为 None，则对 pred, gt 进行概率阈值分割：>threshold 视为 1。

    返回
    ----
    float
        ASSD 值（单位同 spacing）。
    """
    # 去掉多余维度，例如 [1,1,128,128,128] -> [128,128,128]
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, gt {gt.shape}")

    # 若给定阈值，则将概率图二值化
    if threshold is not None:
        pred = (pred > threshold).astype(np.uint8)
        gt = (gt > threshold).astype(np.uint8)

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 若两者都没有前景，ASSD 定义可以为 0
    if not pred.any() and not gt.any():
        return 0.0
    # 若只有一方有前景，ASSD 可认为是无穷大或给一个较大值
    if pred.any() and not gt.any():
        return np.inf
    if gt.any() and not pred.any():
        return np.inf

    # 提取表面
    pred_surface = extract_surface(pred)
    gt_surface = extract_surface(gt)

    pred_surface_pts = np.array(np.where(pred_surface)).T  # [Np, 3] -> (z,y,x)
    gt_surface_pts = np.array(np.where(gt_surface)).T      # [Ng, 3]

    if pred_surface_pts.size == 0 or gt_surface_pts.size == 0:
        # 万一没有提取到表面（例如全体都是1/0的极端情况），退化为整体距离
        pred_surface_pts = np.array(np.where(pred)).T
        gt_surface_pts = np.array(np.where(gt)).T

    # 距离变换：在对方的反掩膜上计算距离
    # distance_transform_edt 返回到非零像素的距离，因此我们对 ~mask 求 EDT
    # 注意：缩放因子 sampling 对应 spacing
    # edt_input 为 0 的位置被视为目标点(前景)，非 0 为背景
    # 所以这里要传入 ~mask (background 为 True) 或反过来？
    # 标准做法：对 mask==0 求 EDT，得到到“前景(非0)”的距离。
    # 因此在这里我们用 1 - mask（或 ~mask），并指定 sampling=spacing。
    pred_dt = ndimage.distance_transform_edt(~pred, sampling=spacing)
    gt_dt = ndimage.distance_transform_edt(~gt, sampling=spacing)

    # 在对方表面点取距离
    # pred → gt：在 gt 的表面点上看 pred_dt
    distances_pred_to_gt = pred_dt[gt_surface_pts[:, 0],
                                   gt_surface_pts[:, 1],
                                   gt_surface_pts[:, 2]]
    # gt → pred：在 pred 的表面点上看 gt_dt
    distances_gt_to_pred = gt_dt[pred_surface_pts[:, 0],
                                 pred_surface_pts[:, 1],
                                 pred_surface_pts[:, 2]]

    assd = (distances_pred_to_gt.mean() + distances_gt_to_pred.mean()) / 2.0
    return float(assd)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out
