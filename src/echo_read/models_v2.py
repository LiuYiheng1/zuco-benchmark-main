import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(torch.tensor(alpha))
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class EEGEncoderSharedPrivate(nn.Module):
    def __init__(self, input_dim=420, d_shared=32, d_private=32, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc_shared = nn.Linear(128, d_shared)
        self.fc_private = nn.Linear(128, d_private)
    
    def forward(self, x):
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        z_shared = self.fc_shared(x)
        z_private = self.fc_private(x)
        return z_shared, z_private


class GazeEncoderSharedPrivate(nn.Module):
    def __init__(self, input_dim=9, d_shared=32, d_private=32, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.fc_shared = nn.Linear(64, d_shared)
        self.fc_private = nn.Linear(64, d_private)
    
    def forward(self, x):
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        z_shared = self.fc_shared(x)
        z_private = self.fc_private(x)
        return z_shared, z_private


class AgreementBottleneck(nn.Module):
    def __init__(self, d_shared=32, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        input_dim = d_shared * 4 + 1  
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, d_shared)
    
    def forward(self, z_e_shared, z_g_shared):
        z_agree = z_e_shared * z_g_shared
        z_diff = torch.abs(z_e_shared - z_g_shared)
        
        norm_e = F.normalize(z_e_shared, dim=1)
        norm_g = F.normalize(z_g_shared, dim=1)
        cos_sim = torch.sum(norm_e * norm_g, dim=1, keepdim=True)
        
        h = torch.cat([z_e_shared, z_g_shared, z_agree, z_diff, cos_sim], dim=1)
        h = self.drop1(F.gelu(self.ln1(self.fc1(h))))
        z_shared = self.fc2(h)
        
        return z_shared, z_agree, z_diff, cos_sim.squeeze(1)


class ClassifierV2(nn.Module):
    def __init__(self, d_shared=32, num_classes=2, dropout_rate=0.1):
        super().__init__()
        input_dim = d_shared * 2 + 1 + d_shared  
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, z_g_shared, z_shared, z_diff, cos_sim):
        h = torch.cat([z_g_shared, z_shared, z_diff, cos_sim.unsqueeze(1)], dim=1)
        h = self.drop1(F.gelu(self.ln1(self.fc1(h))))
        logits = self.fc2(h)
        return logits


class SubjectClassifier(nn.Module):
    def __init__(self, input_dim=32, num_subjects=16, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_subjects)
    
    def forward(self, z, alpha=1.0):
        z = GradientReversal.apply(z, alpha)
        h = self.drop1(F.gelu(self.ln1(self.fc1(z))))
        logits = self.fc2(h)
        return logits


class EchoV2SharedOnly(nn.Module):
    def __init__(self, d_shared=32, dropout_rate=0.1):
        super().__init__()
        self.eeg_encoder = EEGEncoderSharedPrivate(
            input_dim=420, d_shared=d_shared, d_private=0, dropout_rate=dropout_rate
        )
        self.gaze_encoder = GazeEncoderSharedPrivate(
            input_dim=9, d_shared=d_shared, d_private=0, dropout_rate=dropout_rate
        )
        self.agreement_bottleneck = AgreementBottleneck(
            d_shared=d_shared, hidden_dim=64, dropout_rate=dropout_rate
        )
        self.classifier = ClassifierV2(d_shared=d_shared, num_classes=2, dropout_rate=dropout_rate)
    
    def forward(self, batch):
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        
        z_e_shared, _ = self.eeg_encoder(eeg)
        z_g_shared, _ = self.gaze_encoder(gaze)
        
        z_shared, z_agree, z_diff, cos_sim = self.agreement_bottleneck(z_e_shared, z_g_shared)
        logits = self.classifier(z_g_shared, z_shared, z_diff, cos_sim)
        
        return {
            "logits": logits,
            "z_e_shared": z_e_shared,
            "z_g_shared": z_g_shared,
            "z_shared": z_shared,
            "z_agree": z_agree,
            "z_diff": z_diff,
            "cos_sim": cos_sim,
            "z_e_private": None,
            "z_g_private": None
        }


class EchoV2SharedPrivate(nn.Module):
    def __init__(self, d_shared=32, d_private=32, dropout_rate=0.1):
        super().__init__()
        self.d_shared = d_shared
        self.d_private = d_private
        
        self.eeg_encoder = EEGEncoderSharedPrivate(
            input_dim=420, d_shared=d_shared, d_private=d_private, dropout_rate=dropout_rate
        )
        self.gaze_encoder = GazeEncoderSharedPrivate(
            input_dim=9, d_shared=d_shared, d_private=d_private, dropout_rate=dropout_rate
        )
        self.agreement_bottleneck = AgreementBottleneck(
            d_shared=d_shared, hidden_dim=64, dropout_rate=dropout_rate
        )
        self.classifier = ClassifierV2(d_shared=d_shared, num_classes=2, dropout_rate=dropout_rate)
    
    def forward(self, batch):
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        
        z_e_shared, z_e_private = self.eeg_encoder(eeg)
        z_g_shared, z_g_private = self.gaze_encoder(gaze)
        
        z_shared, z_agree, z_diff, cos_sim = self.agreement_bottleneck(z_e_shared, z_g_shared)
        logits = self.classifier(z_g_shared, z_shared, z_diff, cos_sim)
        
        return {
            "logits": logits,
            "z_e_shared": z_e_shared,
            "z_g_shared": z_g_shared,
            "z_shared": z_shared,
            "z_agree": z_agree,
            "z_diff": z_diff,
            "cos_sim": cos_sim,
            "z_e_private": z_e_private,
            "z_g_private": z_g_private
        }


class EchoV2Full(nn.Module):
    def __init__(self, d_shared=32, d_private=32, num_subjects=16, dropout_rate=0.1):
        super().__init__()
        self.d_shared = d_shared
        self.d_private = d_private
        
        self.eeg_encoder = EEGEncoderSharedPrivate(
            input_dim=420, d_shared=d_shared, d_private=d_private, dropout_rate=dropout_rate
        )
        self.gaze_encoder = GazeEncoderSharedPrivate(
            input_dim=9, d_shared=d_shared, d_private=d_private, dropout_rate=dropout_rate
        )
        self.agreement_bottleneck = AgreementBottleneck(
            d_shared=d_shared, hidden_dim=64, dropout_rate=dropout_rate
        )
        self.classifier = ClassifierV2(d_shared=d_shared, num_classes=2, dropout_rate=dropout_rate)
        self.subject_classifier = SubjectClassifier(
            input_dim=d_shared, num_subjects=num_subjects, dropout_rate=dropout_rate
        )
    
    def forward(self, batch, grl_alpha=1.0):
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        
        z_e_shared, z_e_private = self.eeg_encoder(eeg)
        z_g_shared, z_g_private = self.gaze_encoder(gaze)
        
        z_shared, z_agree, z_diff, cos_sim = self.agreement_bottleneck(z_e_shared, z_g_shared)
        logits = self.classifier(z_g_shared, z_shared, z_diff, cos_sim)
        subject_logits = self.subject_classifier(z_shared, alpha=grl_alpha)
        
        return {
            "logits": logits,
            "subject_logits": subject_logits,
            "z_e_shared": z_e_shared,
            "z_g_shared": z_g_shared,
            "z_shared": z_shared,
            "z_agree": z_agree,
            "z_diff": z_diff,
            "cos_sim": cos_sim,
            "z_e_private": z_e_private,
            "z_g_private": z_g_private
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary_v2(model):
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append("-" * 50)
    
    if hasattr(model, "eeg_encoder"):
        summary.append(f"EEGEncoderSharedPrivate: {count_parameters(model.eeg_encoder):,} params")
    if hasattr(model, "gaze_encoder"):
        summary.append(f"GazeEncoderSharedPrivate: {count_parameters(model.gaze_encoder):,} params")
    if hasattr(model, "agreement_bottleneck"):
        summary.append(f"AgreementBottleneck: {count_parameters(model.agreement_bottleneck):,} params")
    if hasattr(model, "classifier"):
        summary.append(f"Classifier: {count_parameters(model.classifier):,} params")
    if hasattr(model, "subject_classifier"):
        summary.append(f"SubjectClassifier: {count_parameters(model.subject_classifier):,} params")
    
    return "\n".join(summary)