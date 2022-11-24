from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from imageio import imwrite
from function import adaptive_instance_normalization, coral, calc_mean_std
import net_ada_assignment_v2 as net
import os
import numpy as np
import time

class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none', feat_dim=512):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.A = nn.Parameter(torch.eye(feat_dim), requires_grad=True)

    def forward(self, x, y, eps=None):
        if eps is None:
            this_eps = self.eps
        else:
            this_eps = eps
        print(this_eps)
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False, device=x.device).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                     requires_grad=False, device=y.device).fill_(1.0 / y_points).squeeze()


        u = torch.zeros_like(mu, device=mu.device)
        v = torch.zeros_like(nu, device=nu.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = this_eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v, this_eps), dim=-1)) + u
            v = this_eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v, this_eps).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break
        # print(f"Converge in {actual_nits}")

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V, this_eps))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return (cost, mu, nu), pi, C

    def M(self, C, u, v, eps):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / eps

    def _cost_matrix(self, x, y):
        "Returns the matrix of cosine distance."
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = y / torch.norm(y, dim=-1, keepdim=True)
        C = 1 - torch.bmm(x, y.permute(0, 2, 1))
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    
class Net(nn.Module):
    def __init__(self, encoder, decoder, encoder_trainable=True, metric_learning=True, eps=1e-2, hungarian=0, alpha_pred_version=1):
        print('INFO: ada assignment')
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        if alpha_pred_version == 1:
            self.alpha_predictor = nn.Sequential(
                nn.Conv2d(2*512, 256, 3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, 3, padding=1),
            )
        elif alpha_pred_version == 2:
            self.alpha_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(2*512, 256, 1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, 1),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 1),
            )
        elif alpha_pred_version == 3:
            self.alpha_predictor = nn.Sequential(
                nn.Conv2d(2*512, 256, 3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(True),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 512, 3, padding=1),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
        elif alpha_pred_version == 4 or alpha_pred_version == 5:
            # 1x1 prediction
            self.alpha_predictor = nn.Conv2d(1, 1, 1, padding=0)
            
        elif alpha_pred_version == 6: # redo of version 4
            self.alpha_predictor = nn.Sequential(
                nn.Conv2d(1, 1, 1, padding=0),
                nn.ReLU(True),
            )
        elif alpha_pred_version == 7: # redo of version 5
            self.alpha_predictor = nn.Sequential(
                nn.Conv2d(1, 1, 1, padding=0),
                nn.Sigmoid()
            )

        self.alpha_pred_version = alpha_pred_version


        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=100)
        if not metric_learning:
            self.sinkhorn.eval()

        # fix the encoder if not train encoder
        if not encoder_trainable:
            for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
                for param in getattr(self, name).parameters():
                    param.requires_grad = False
                    
        self.hungarian = hungarian

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def style_assign(self, content_f, style_f, eps):
        bs, dim, h, w = content_f.shape
        content_f_ravel = content_f.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel = style_f.view(bs, dim, -1).permute(0, 2, 1)
        cost, pi, _ = self.sinkhorn(content_f_ravel, style_f_ravel, eps)
        pi = pi * h * w
        if self.hungarian:
            pi = batched_hungarian(pi, is_profit=True)
        feat = torch.bmm(pi, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return feat
    
    def norm_content_feat(self, content_feat):
        c_mean, c_std = calc_mean_std(content_feat)
        c_normed = (content_feat - c_mean) / c_std
        return c_normed

    def forward(self, content, style):
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.style_assign(content_feat, style_feats[-1])
        c_normed = self.norm_content_feat(content_feat)
        if self.alpha_pred_version < 4:
            alpha = self.alpha_predictor(torch.cat([t, content_feat], dim=1))
        elif self.alpha_pred_version >= 4:
            t_ = t / torch.norm(t, dim=1, keepdim=True)
            content_feat_ = content_feat / torch.norm(content_feat, dim=1, keepdim=True)
            feat_simi = torch.sum(t_ * content_feat_, dim=1, keepdim=True)
            if self.alpha_pred_version == 4 or self.alpha_pred_version == 6:
                alpha = self.alpha_predictor(feat_simi)
            elif self.alpha_pred_version == 5 or self.alpha_pred_version == 7:
                _, s_std = calc_mean_std(style_feats[-1])
                alpha = self.alpha_predictor(feat_simi)
                alpha = alpha * s_std
        else:
            raise NotImplementedError
        t = alpha * c_normed + t

        g_t = self.decoder(t)
        return g_t
    
    def diversify_transfer(self, content, style, eps=1e-2):
        print(eps)
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.style_assign(content_feat, style_feats[-1], eps=eps)
        c_normed = self.norm_content_feat(content_feat)
        if self.alpha_pred_version < 4:
            alpha = self.alpha_predictor(torch.cat([t, content_feat], dim=1))
        elif self.alpha_pred_version >= 4:
            t_ = t / torch.norm(t, dim=1, keepdim=True)
            content_feat_ = content_feat / torch.norm(content_feat, dim=1, keepdim=True)
            feat_simi = torch.sum(t_ * content_feat_, dim=1, keepdim=True)
            if self.alpha_pred_version == 4 or self.alpha_pred_version == 6:
                alpha = self.alpha_predictor(feat_simi)
            elif self.alpha_pred_version == 5 or self.alpha_pred_version == 7:
                _, s_std = calc_mean_std(style_feats[-1])
                alpha = self.alpha_predictor(feat_simi)
                alpha = alpha * s_std
        else:
            raise NotImplementedError
        t = alpha * c_normed + t

        g_t = self.decoder(t)
        return g_t


def test_transform(size, crop=False):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop((size, size)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

    
def style_assignment(model, content, style, eps):
    output = model.diversify_transfer(content, style, eps)
    output = output.cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='../style_transfer_qualitative_result/coco_pbn_5000/style_assign_trained')
    parser.add_argument('--input_file', type=str, default='coco_pbn_5000_train_image_pairs.txt')
    parser.add_argument('--eps', type=float, default=1e-2)
    parser.add_argument('--repeat_times', type=int, default=5)
    args = parser.parse_args()
    alpha_pred_version = 7
    
    os.makedirs(args.out_dir, exist_ok=True)

    content_size = 512
    style_size = 512
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = net.get_decoder()
    vgg = net.get_vgg()
    model = Net(vgg, decoder, eps=1e-2, alpha_pred_version=alpha_pred_version).to(device)
    model_path = f'experiments/AdaAssignV2-PT_1-ENC_1-ML_0-EPS_{args.eps}-AP_{alpha_pred_version}/iter_160000.pth.tar'

    weights = torch.load(model_path)['model']
    weights = {k.replace('module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights)

    model.eval()
    
    content_tf = test_transform(content_size)
    style_tf = test_transform(style_size)
    
    with open(args.input_file, 'r') as IN:
        for idx, l in enumerate(IN):
            content_dir, style_dir, content_path, style_path, out_path = l.strip().split(',')
            content_path = os.path.join(content_dir, content_path)
            style_path = os.path.join(style_dir, style_path)
            
            content = content_tf(Image.open(str(content_path)).convert('RGB')).unsqueeze(0).to(device)
            style = style_tf(Image.open(str(style_path)).convert('RGB')).unsqueeze(0).to(device)
            
            for repeat in range(args.repeat_times):
                this_out_path = out_path.split('.')[0] + '_{}'.format(repeat+1) + '.jpg'
                this_eps = 10**(-repeat)
                with torch.no_grad():
                    out = style_assignment(model, content, style, this_eps)
                imwrite(
                    os.path.join(args.out_dir, this_out_path),
                    out
                )
            print(f'Complete {idx+1}')