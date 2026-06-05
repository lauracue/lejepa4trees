# -*- coding: utf-8 -*-
"""
Created on Thu May 28 12:47:17 2026

@author: cuel001
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import timm, wandb, hydra, tqdm
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP
from sklearn.metrics import f1_score
from dataloader import RandomRasterCropDataset, JPGDataset, UnifiedDataset
from torch.utils.data import ConcatDataset

from pathlib import Path

class SIGReg(torch.nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, proj_dim=128, crop_size=128):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",  # "vit_small_patch8_224"
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            # img_size=crop_size,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)   
    

@hydra.main(version_base=None)
def main(cfg: DictConfig):
    
    RASTERS_FOLDER = ["./data/2023/ortos_2023-11",
                      "./data/2025/ortos_2025-11",
                      "./data/2025/ortos_2025-12"]
    
    TEST_FOLDER = "./data/correcting_geospatial_data/cropped_images"
    
    jpg_files = list(Path(TEST_FOLDER).glob("*.jpg"))

    raster_files = []
    for pth in RASTERS_FOLDER:
        raster_files.extend(Path(pth).glob("*.tif"))

    print(f"Found {len(raster_files)} rasters")
    
    wandb.init(project="LeJEPA", config=dict(cfg))
    torch.manual_seed(0)

    # -----------------------------
    # UNLABELED SSL DATA
    # -----------------------------
    train_ssl_ds = RandomRasterCropDataset(
        raster_files=raster_files,
        crop_size=cfg.crop_size,
        views=cfg.V,
        samples=cfg.samples
    )
    

    dataset = JPGDataset(
        jpg_files=jpg_files,
        crop_size=cfg.crop_size
    )

    # split into train/test
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(0)
    )

    train_labeled = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    test = DataLoader(
        test_ds,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=0,
    )
    
    train = DataLoader(
        train_ssl_ds,
        batch_size=cfg.bs,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    


    # modules and loss
    net = ViTEncoder(proj_dim=cfg.proj_dim).to("cuda")
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, cfg.num_classes)).to("cuda")
    sigreg = SIGReg().to("cuda")
    # Optimizer and scheduler
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt_ssl = torch.optim.AdamW([g1])
    opt_lab = torch.optim.AdamW([g2])
    warmup_steps = len(train)
    total_steps = len(train) * cfg.epochs
    s1 = LinearLR(opt_ssl, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt_ssl, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler_ssl = SequentialLR(opt_ssl, schedulers=[s1, s2], milestones=[warmup_steps])
    
    s3 = LinearLR(opt_lab, start_factor=0.01, total_iters=warmup_steps)
    s4 = CosineAnnealingLR(opt_lab, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler_lab = SequentialLR(opt_lab, schedulers=[s3,s4], milestones=[warmup_steps])

    scaler_ssl = GradScaler(enabled="cuda" == "cuda")
    scaler_lab = GradScaler(enabled="cuda" == "cuda")
    

    # Training
    for epoch in range(cfg.epochs):
        net.train(), probe.train()
        for vs in tqdm.tqdm(train, total=len(train)):
            with autocast("cuda", dtype=torch.bfloat16):
                vs = vs.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                
    
            opt_ssl.zero_grad()
            scaler_ssl.scale(lejepa_loss).backward()
            scaler_ssl.step(opt_ssl)
            scaler_ssl.update()
            scheduler_ssl.step()
            wandb.log(
                {
                    "train/lejepa": lejepa_loss.item(),
                    "train/sigreg": sigreg_loss.item(),
                    "train/inv": inv_loss.item(),
                }
            )
            
        running_loss = 0.0
        running_correct = 0
        running_total = 0
            
        for vl, y in tqdm.tqdm(train_labeled, total=len(train_labeled)):
            with autocast("cuda", dtype=torch.bfloat16):
                vl = vl.to("cuda", non_blocking=True)
                y = y.to("cuda", non_blocking=True)
                emb, proj = net(vl)
                
                logits = probe(emb.detach())

                loss = F.cross_entropy(logits, y)

            opt_lab.zero_grad()
            scaler_lab.scale(loss).backward()
            scaler_lab.step(opt_lab)
            scaler_lab.update()            
            scheduler_lab.step()
            
            # metrics
            preds = logits.argmax(1)

            running_correct += (preds == y).sum().item()
            running_total += y.size(0)

            running_loss += loss.item()

            wandb.log({
                "train_labeled/loss": loss.item(),
            })

        train_acc = running_correct / running_total
        train_loss = running_loss / len(train_labeled)

        # Evaluation
        net.eval(), probe.eval()

        correct = 0
        total = 0
        test_loss = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.inference_mode():

            for vs, y in tqdm.tqdm(test, total=len(test)):
                with autocast("cuda", dtype=torch.bfloat16):
                    vs = vs.to("cuda", non_blocking=True)
                    y = y.to("cuda", non_blocking=True)        
                    emb, _ = net(vs)
                    logits = probe(emb)
                    loss = F.cross_entropy(logits, y)
        
                preds = logits.argmax(1)
        
                correct += (preds == y).sum().item()
        
                total += y.size(0)
        
                test_loss += loss.item()
        
                # store predictions for F1
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        test_acc = correct / total
        test_loss = test_loss / len(test)
        

        per_class_f1 = f1_score(
            all_targets,
            all_preds,
            average=None,
            labels=list(range(cfg.num_classes)),
            zero_division=0,
        )
        
        macro_f1 = f1_score(
            all_targets,
            all_preds,
            average="macro",
            zero_division=0,
        )
        
        weighted_f1 = f1_score(
            all_targets,
            all_preds,
            average="weighted",
            zero_division=0,
        )
        
        # ---------------------------------------------------
        # PRINT METRICS
        # ---------------------------------------------------
        
        print(
            f"Epoch {epoch} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f} | "
            f"macro_f1={macro_f1:.4f}"
        )
        
        print("Per-class F1 scores:")
        
        for cls_idx, f1 in enumerate(per_class_f1):
            print(f"Class {cls_idx}: F1 = {f1:.4f}")
        
        # ---------------------------------------------------
        # WANDB LOGGING
        # ---------------------------------------------------
        
        log_dict = {
            "epoch": epoch,
            "train_labeled/acc": train_acc,
            "test/acc": test_acc,
            "test/loss": test_loss,
            "test/f1_macro": macro_f1,
            "test/f1_weighted": weighted_f1,
        }
        
        # add per-class f1
        for cls_idx, f1 in enumerate(per_class_f1):
            log_dict[f"test/f1_class_{cls_idx}"] = f1
        
        wandb.log(log_dict)


if __name__ == "__main__":
    cfg = OmegaConf.create({
                "bs": 256,
                "lr": 2e-3,
                "lamb": 0.02, 
                "V": 4,
                "proj_dim": 16,
                "epochs": 800,
                "crop_size": 128,
                "samples": 10000,
                "num_classes": 20,
            })
    main(cfg)
