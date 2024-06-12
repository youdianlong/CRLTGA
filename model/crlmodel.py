import os

import torch
from torch import nn
from torchvision import models

import utils
from net.generator import Generator
import torch.nn.functional as F

from scm import SCM

# from causal.NodeSpecificSCM import SCM
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


class TGAVAE(nn.Module):

    def __init__(self, args, causal_dim, A, model_name):
        super().__init__()
        self.z_dim = args.z_dim
        self.causal_dim = causal_dim
        self.A = A
        self.model_name = model_name
        self.alpha = args.alpha
        self.beta = args.beta
        self.gama = args.gamma
        self.save_dir = os.path.join('checkpoints', model_name)
        self.encoder = models.resnet50(pretrained=False)
        pretrained_dict = torch.load('/data0/sjw/code/triplet-causal-attention-vae/resnet50-0676ba61.pth')  # 加载本地目录中的预训练文件
        self.encoder.load_state_dict(pretrained_dict)  # 将预训练权重赋值给resnet50模型
        for param in self.encoder.parameters():
            param.requires_grad = True
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                        nn.ReLU())
        self.mean_fc = nn.Linear(1024, args.z_dim)
        self.logvar_fc = nn.Linear(1024, args.z_dim)
        self.decoder = Generator(latent_dim=args.z_dim, image_size=args.image_size)

        self.scm = SCM(causal_dim, A)

    def reparameterize(self, mu, v):
        mu = mu.to(device)
        v = v.to(device)
        # 64*3*4
        sample = torch.randn(mu.size()).to(device)
        z = mu + (v ** 0.5) * sample
        return z

    def encode(self, x):
        z_params = self.encoder(x.to(device))

        return F.relu(self.mean_fc(z_params)), F.softplus(F.relu(self.logvar_fc(z_params))) + 1e-8

    def forward(self, data, label, mode='triplet'):
        eps, var = self.encode(data)

        causal, eps_loss, A = self.scm(eps[:, :self.causal_dim])

        m = torch.cat([causal, eps[:, self.causal_dim:]], dim=1)

        z = self.reparameterize(m, 0.01 * var)

        a_hat = self.decoder(z)

        mse_loss = F.mse_loss(data, a_hat)
        kl = self.kl_loss(m, var, label)
        kl = kl.mean()

        params = {}
        params['z'] = z
        params['causal'] = causal
        params['rec'] = a_hat

        loss = mse_loss + kl

        loss_dic = {}
        loss_dic['mse'] = mse_loss.item()
        loss_dic['kl'] = kl.item()
        loss_dic['loss'] = loss.item()

        loss_sub = {}
        loss_sub['mse'] = mse_loss
        loss_sub['kl'] = kl

        return loss, loss_dic, params, loss_sub

    def decode(self, causal_m, m, var):
        m = torch.cat([causal_m, m[:, self.causal_dim:]], dim=1)
        # m = self.fc(m)
        z = self.reparameterize(m, 0.001 * var)
        return self.decoder(z)

    def kl_loss(self, m, var, label):
        kl = utils.kl_normal(m[:, :self.causal_dim], var[:, :self.causal_dim], label,
                             torch.ones_like(var[:, :self.causal_dim], device=device))
        kl += 0.1 * utils.kl_normal(m[:, self.causal_dim:], var[:, self.causal_dim:],
                                    torch.zeros_like(m[:, self.causal_dim:], device=device),
                                    torch.ones_like(var[:, self.causal_dim:], device=device))
        return kl

    def triplet_loss(self, a, p, n):
        return self.gama * F.triplet_margin_loss(a, p, n, margin=0.01, p=2)

    def decode(self, x):
        return self.decoder(x)
