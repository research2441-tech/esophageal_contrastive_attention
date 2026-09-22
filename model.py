from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden=max(channels//reduction,8)
        self.mlp=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,hidden,1),
                               nn.ReLU(inplace=True),nn.Conv2d(hidden,channels,1),nn.Sigmoid())
        self.spatial=nn.Sequential(nn.Conv2d(2,1,7,padding=3,bias=False),nn.Sigmoid())
    def forward(self,x):
        x=x*self.mlp(x)
        avg=torch.mean(x,1,keepdim=True); mx=torch.amax(x,1,keepdim=True)
        return x*self.spatial(torch.cat([avg,mx],1))

class ContrastiveAttentionNet(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True, attention=True,
                 embedding_dim=128, projection_dim=128, dropout=.3, num_classes=2):
        super().__init__()
        self.encoder=timm.create_model(backbone,pretrained=pretrained,features_only=True,out_indices=(-1,))
        ch=self.encoder.feature_info.channels()[-1]
        self.attn=ChannelSpatialAttention(ch) if attention else nn.Identity()
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.embedding=nn.Sequential(nn.Flatten(),nn.Dropout(dropout),nn.Linear(ch,embedding_dim),
                                     nn.BatchNorm1d(embedding_dim),nn.ReLU(inplace=True))
        self.projector=nn.Sequential(nn.Linear(embedding_dim,projection_dim),nn.ReLU(inplace=True),
                                     nn.Linear(projection_dim,projection_dim))
        self.classifier=nn.Linear(embedding_dim,num_classes)
    def forward(self,x):
        f=self.encoder(x)[0]; f=self.attn(f); z=self.embedding(self.pool(f))
        logits=self.classifier(z)
        proj=F.normalize(self.projector(z),dim=1)
        return {"logits":logits,"embedding":z,"projection":proj}

def supervised_contrastive_loss(features, labels, temperature=0.1):
    features=F.normalize(features,dim=1)
    sim=(features@features.T)/temperature
    n=len(labels); eye=torch.eye(n,device=features.device,dtype=torch.bool)
    sim=sim.masked_fill(eye,-1e9)
    labels=labels.view(-1,1)
    pos=(labels==labels.T) & ~eye
    logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
    denom=pos.sum(1).clamp_min(1)
    loss=-(logp*pos).sum(1)/denom
    return loss.mean()

def build_model(cfg):
    m=cfg["model"]
    return ContrastiveAttentionNet(m["backbone"],m["pretrained"],m["attention"],
                                   m["embedding_dim"],m["projection_dim"],
                                   m["dropout"],m["num_classes"])
