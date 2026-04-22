import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from model.resnet import Deep_Vision_Feature_Model


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class fusionLayer(nn.Module):
    def __init__(self, in_channel, outChans, depth, act):
        super(fusionLayer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        m_batchsize, C, depth, height, width = x1.size()
        fusion = self.sigmoid((x1+x2))
        # proj_value = x1.view(m_batchsize, C, -1)
        # out = torch.bmm(attention, proj_value)
        out = fusion.view(m_batchsize, C, depth, height, width)
        return out
class SCSELayer(nn.Module):
    def __init__(self, channel=32, reduction=8):
        super(SCSELayer, self).__init__()
        self.cse_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.cse_fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.sse_conv = nn.Conv3d(channel, 1, 1, padding=0)

    def forward(self, x):
        b, c, z, w, h = x.size()
        cse_y = self.cse_avg_pool(x).view(b, c)
        cse_y = self.cse_fc(cse_y).view(b, c, 1, 1, 1)
        sse_y = self.sse_conv(x)

        return x * cse_y.expand_as(x) + x * sse_y.expand_as(x)

class MIA_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(MIA_Module, self).__init__()
        self.chanel_in = in_dim
        self.conv1 = nn.Conv3d(in_channels=512, out_channels=14, kernel_size=3, stride=1, padding=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X z*y*x)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # img_F_ds = self.conv1(x)
        # img_F_ds = F.interpolate(img_F_ds, size=(192, 64, 32), mode='trilinear', align_corners=True)
        ### MIA module #######
        m_batchsize, C, depth, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, depth, height, width)
        img_F_mia = self.gamma * out + x

        # img_F_mia_fusion = self.conv1(img_F_mia)
        # img_F_mia_fusion = F.interpolate(img_F_mia_fusion, size=(192, 64, 32), mode='trilinear', align_corners=True)

        return img_F_mia

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out

class AttentionGateFusion(nn.Module):
    def __init__(self, in_channels_A=2048, in_channels_B=192, out_channels=512):
        super().__init__()
        self.reduce_A = nn.Conv3d(in_channels_A, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, A, B):
        A_proj = self.reduce_A(A)
        concat = torch.cat([A_proj, B], dim=1)  # [1, 384, 16, 16, 16]
        attention = self.gate(concat)  # [1, 192, 16, 16, 16]
        fused = attention * A_proj + (1 - attention) * B
        return fused
class MultimodalCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultimodalCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x1_feature, x2_feature):
        """
        x1_feature: [B, C, D, H, W]
        x2_feature: [B, C, D, H, W]
        """
        B, C, D, H, W = x1_feature.shape

        # Flatten spatial dims: [B, C, N] -> transpose -> [B, N, C]
        x1_flat = x1_feature.view(B, C, -1).transpose(1, 2)  # [B, N, C]
        x2_flat = x2_feature.view(B, C, -1).transpose(1, 2)  # [B, N, C]

        # Apply cross attention: Q from x1, K/V from x2
        fused, _ = self.attention(query=x1_flat, key=x2_flat, value=x2_flat)

        # Reshape back to [B, C, D, H, W]
        fused = fused.transpose(1, 2).view(B, C, D, H, W)

        return fused
class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class mm_Segmentation_Foundation_Model(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, use_text_prompt, n_class=9, act='relu'):
        super(mm_Segmentation_Foundation_Model, self).__init__()
        self.text_prompt = use_text_prompt

        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, n_class)
        self.out = UnetOutBlock(spatial_dims=3, in_channels=64, out_channels=n_class)

        self.MIA_module = MIA_Module(9)
        self.fusion_layer =fusionLayer(512,512, 1,act='relu')

        self.DVF_branch = Deep_Vision_Feature_Model(model_depth=152, n_classes=256, input_W=96, input_H=96, input_D=96)

        self.multimoal_fusion_module = MultimodalCrossAttention(embed_dim=2048, num_heads=4)
        self.feat_fusion = AttentionGateFusion()

    def forward(self, x1):## x1, x2 are two different modality data, such as CT and MRI
        ## encode modality data from x1
        self.out64_x1, self.skip_out64_x1 = self.down_tr64(x1)
        self.out128_x1,self.skip_out128_x1 = self.down_tr128(self.out64_x1)
        self.out256_x1,self.skip_out256_x1 = self.down_tr256(self.out128_x1)
        self.out512_x1,self.skip_out512_x1 = self.down_tr512(self.out256_x1)
        _, x1_deep_feat = self.DVF_branch(x1)
        feat_fusion_x1 = self.feat_fusion(x1_deep_feat, self.out512_x1) #

        if self.text_prompt:
            # print("use text prompt!")
            if self.encoding == 'rand_embedding':
                task_encoding = self.organ_embedding.weight.unsqueeze(3).unsqueeze(3).unsqueeze(3)
            elif self.encoding == 'word_embedding':
                # for modality t1
                x1_task_encoding = F.relu(self.text_to_vision(self.organ_embedding[0]))
                x1_task_encoding = x1_task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
                # for modality t1c
                x2_task_encoding = F.relu(self.text_to_vision(self.organ_embedding[1]))  # bert: [2,14, 768] CLIP: [2,15,512]
                x2_task_encoding = x2_task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)

            x1_feat = self.GAP(self.out512_x1)
            x2_feat = self.GAP(x1_deep_feat)
            b = x1_feat.shape[0]
            x1_logits_array = []
            x2_logits_array = []
            for i in range(b):
                x1_cond = torch.cat([x1_feat[i].unsqueeze(0).repeat(self.class_num, 1, 1, 1, 1), x1_task_encoding], 1)
                x1_params = self.controller(x1_cond)
                x1_params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
                x1_head_inputs = self.precls_conv(x1_deep_feat[i].unsqueeze(0))
                x1_head_inputs = x1_head_inputs.repeat(self.class_num, 1, 1, 1, 1)
                N, _, D, H, W = x1_head_inputs.size()
                x1_head_inputs = x1_head_inputs.reshape(1, -1, D, H, W)
                weights, biases = self.parse_dynamic_params(x1_params, 8, self.weight_nums, self.bias_nums)
                x1_logits = self.heads_forward(x1_head_inputs, weights, biases, N)
                x1_logits_array.append(x1_logits.reshape(1, -1, D, H, W))

                # for modality x2
                x2_cond = torch.cat([x2_feat[i].unsqueeze(0).repeat(self.class_num, 1, 1, 1, 1), x2_task_encoding], 1)
                x2_params = self.controller(x2_cond)
                x2_params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
                x2_head_inputs = self.precls_conv(feat_fusion_x1[i].unsqueeze(0))
                x2_head_inputs = x2_head_inputs.repeat(self.class_num, 1, 1, 1, 1)
                N, _, D, H, W = x2_head_inputs.size()
                x2_head_inputs = x2_head_inputs.reshape(1, -1, D, H, W)
                weights, biases = self.parse_dynamic_params(x2_params, 8, self.weight_nums, self.bias_nums)
                x2_logits = self.heads_forward(x2_head_inputs, weights, biases, N)
                x2_logits_array.append(x2_logits.reshape(1, -1, D, H, W))

            x1_seg_out = torch.cat(x1_logits_array, dim=0)
            x2_seg_out = torch.cat(x2_logits_array, dim=0)
            fusion_feature = self.multimoal_fusion_module(x1_seg_out,x2_seg_out)
        else:
            out_fuse = feat_fusion_x1

        ## decode for modality data from x1
        self.out_up_256_x1 = self.up_tr256(feat_fusion_x1,self.skip_out256_x1)
        self.out_up_128_x1 = self.up_tr128(self.out_up_256_x1, self.skip_out128_x1)
        self.out_up_64_x1 = self.up_tr64(self.out_up_128_x1, self.skip_out64_x1)
        # self.out_x1 = self.out_tr(self.out_up_64_x1)
        self.out_x1 = self.out(self.out_up_64_x1)

        # fused_outputs = torch.cat((img_F_ds, img_F_mia_fusion, self.out_x1), dim=1)  # 通道维度拼接
        # fused_outputs = torch.cat((self.out_x1, self.out_x1, self.out_x1), dim=1)  # 通道维度拼接

        return self.out_x1

