import torch
import torch.nn as nn
import numpy as np
from function import SinkhornDistance
from function import calc_mean_std
import scipy.optimize as opt

def get_decoder():
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    return decoder

def get_vgg():
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    return vgg

def batched_hungarian(m, is_profit=False):
    perm_mat_batch = []
    for m_ in m:
        perm_mat_batch.append(hungarian(m_, is_profit))
    return torch.stack(perm_mat_batch, dim=0)

def hungarian(m, is_profit=False):
    '''
    input: a cost matrix or a profit matrix. Size should be (H x W)
    it supports both torch.tensor and numpy.ndarray
    '''
    if isinstance(m, torch.Tensor):
        is_tensor = True
        device = m.device
        m = m.detach().cpu().numpy()
    else:
        is_tensor = False
    if is_profit:
        m = 1 - m
    row, col = opt.linear_sum_assignment(m)
    perm_mat = np.zeros_like(m)
    perm_mat[row, col] = 1
    if is_tensor:
        perm_mat = torch.from_numpy(perm_mat).to(device)
    return perm_mat


class Net(nn.Module):
    def __init__(self, encoder, decoder, encoder_trainable=True, metric_learning=True, eps=1e-2, hungarian=0, dynamic_sinkhorn_weight=0):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=100, dynamic_sinkhorn_weight=dynamic_sinkhorn_weight)
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

    def style_assign(self, content_f, style_f):
        bs, dim, h, w = content_f.shape
        content_f_ravel = content_f.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel = style_f.view(bs, dim, -1).permute(0, 2, 1)
        cost, pi, _ = self.sinkhorn(content_f_ravel, style_f_ravel)
        pi = pi * h * w
        if self.hungarian:
            pi = batched_hungarian(pi, is_profit=True)
        feat = torch.bmm(pi, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return feat

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = self.style_assign(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        return g_t
    
    def debug_style_assign(self, content, style):
        content_feat = self.encode(content)
        bs, dim, h, w = content_feat.shape
        style_feat = self.encode(style)
        _, _, style_h, style_w = style_feat.shape
        content_f_ravel = content_feat.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel = style_feat.view(bs, dim, -1).permute(0, 2, 1)
        (cost, mu, nu), pi, _ = self.sinkhorn.forward(content_f_ravel, style_f_ravel)
        pi = pi*h*w
        if self.hungarian:
            pi = batched_hungarian(pi, is_profit=True)
        feat = torch.bmm(pi, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return self.decoder(feat), pi, mu.view(bs, h, w), nu.view(bs, style_h, style_w)
    
    def debug_style_assign_v2(self, content, style):
        content_feat = self.encode(content)
        bs, dim, h, w = content_feat.shape
        style_feat = self.encode(style)
        _, _, style_h, style_w = style_feat.shape
        content_f_ravel = content_feat.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel = style_feat.view(bs, dim, -1).permute(0, 2, 1)
        (cost, mu, nu), pi, _ = self.sinkhorn.forward(content_f_ravel, style_f_ravel)
        pi = pi*h*w
        max_idx = torch.argmax(pi, dim=-1, keepdim=True)
        binary_pi = torch.zeros_like(pi, device=pi.device).scatter_(2, max_idx, 1.)
        feat = torch.bmm(binary_pi, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return self.decoder(feat), binary_pi, mu.view(bs, h, w), nu.view(bs, style_h, style_w)
    
    def multi_style_assign(self, content, styles):
        content_feat = self.encode(content)
        bs, dim, h, w = content_feat.shape
        style_feat_list = [self.encode(style) for style in styles]
        content_f_ravel = content_feat.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel_list = [style_f.view(bs, dim, -1).permute(0, 2, 1).view(1, -1, dim) for style_f in style_feat_list]
        style_f_ravel = torch.cat(style_f_ravel_list, dim=1)
        cost, pi, _ = self.sinkhorn.forward(content_f_ravel, style_f_ravel)
        print('non bin', (pi*h*w>0.5).sum())
        feat = torch.bmm(pi*h*w, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return self.decoder(feat), pi*h*w
    
    def multi_style_assign_binarize(self, content, styles):
        content_feat = self.encode(content)
        bs, dim, h, w = content_feat.shape
        style_feat_list = [self.encode(style) for style in styles]
        content_f_ravel = content_feat.view(bs, dim, -1).permute(0, 2, 1)
        style_f_ravel_list = [style_f.view(bs, dim, -1).permute(0, 2, 1).view(1, -1, dim) for style_f in style_feat_list]
        style_f_ravel = torch.cat(style_f_ravel_list, dim=1)
        cost, pi, _ = self.sinkhorn.forward(content_f_ravel, style_f_ravel)
        pi = pi * h * w
        max_idx = torch.argmax(pi, dim=-1, keepdim=True)
        binary_pi = torch.zeros_like(pi, device=pi.device).scatter_(2, max_idx, 1.)
        print('bin', binary_pi.sum())
        feat = torch.bmm(binary_pi, style_f_ravel).permute(0, 2, 1).view(bs, dim, h, w)
        return self.decoder(feat), binary_pi


class LossModel(nn.Module):
    def __init__(self, vgg):
        super(LossModel, self).__init__()
        enc_layers = vgg
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        # self.decoder = get_decoder()
        self.mse_loss = nn.MSELoss()
        
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
        
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, content, style, g_t):
        with torch.no_grad():
            content_feat = self.encode(content)
            style_feats = self.encode_with_intermediate(style)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], content_feat)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s