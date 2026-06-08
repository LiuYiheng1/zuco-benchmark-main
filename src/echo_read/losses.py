import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoLoss(nn.Module):
    def __init__(self, lambda_recon=0.05, lambda_cross=0.05, lambda_align=0.01, alpha_eeg=0.05):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_cross = lambda_cross
        self.lambda_align = lambda_align
        self.alpha_eeg = alpha_eeg
    
    def forward(self, outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        
        L_cls = F.cross_entropy(logits, y)
        
        if "gaze_hat_from_zc" in outputs:
            gaze = batch["gaze"]
            eeg = batch["eeg"]
            
            gaze_hat_from_zc = outputs["gaze_hat_from_zc"]
            eeg_hat_from_zc = outputs["eeg_hat_from_zc"]
            gaze_hat_from_e = outputs["gaze_hat_from_e"]
            eeg_hat_from_g = outputs["eeg_hat_from_g"]
            z_e = outputs["z_e"]
            z_g = outputs["z_g"]
            
            L_common_recon = F.mse_loss(gaze_hat_from_zc, gaze) + self.alpha_eeg * F.mse_loss(eeg_hat_from_zc, eeg)
            
            L_cross_pred = F.mse_loss(gaze_hat_from_e, gaze) + self.alpha_eeg * F.mse_loss(eeg_hat_from_g, eeg)
            
            L_latent_align = F.smooth_l1_loss(z_e, z_g)
            
            L = L_cls + self.lambda_recon * L_common_recon + self.lambda_cross * L_cross_pred + self.lambda_align * L_latent_align
            
            loss_dict = {
                "total": L.item(),
                "cls": L_cls.item(),
                "common_recon": L_common_recon.item(),
                "cross_pred": L_cross_pred.item(),
                "latent_align": L_latent_align.item()
            }
        else:
            L = L_cls
            loss_dict = {"total": L.item(), "cls": L_cls.item()}
        
        return L, loss_dict


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, outputs, batch):
        logits = outputs["logits"]
        y = batch["y"]
        L = F.cross_entropy(logits, y)
        return L, {"total": L.item(), "cls": L.item()}