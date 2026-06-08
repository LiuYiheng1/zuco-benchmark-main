import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGObserver(nn.Module):
    def __init__(self, input_dim=420, hidden_dim=32, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, hidden_dim)
    
    def forward(self, x):
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        z_e = self.fc3(x)
        return z_e


class GazeObserver(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, hidden_dim)
    
    def forward(self, x):
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        z_g = self.fc3(x)
        return z_g


class CommonCauseEstimator(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, dropout_rate=0.1):
        super().__init__()
        combined_dim = input_dim * 4
        self.fc1 = nn.Linear(combined_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, hidden_dim)
    
    def forward(self, z_e, z_g):
        diff = torch.abs(z_e - z_g)
        prod = z_e * z_g
        h = torch.cat([z_e, z_g, diff, prod], dim=1)
        h = self.drop1(F.gelu(self.ln1(self.fc1(h))))
        z_c = self.fc2(h)
        return z_c


class GazeDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=9, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, z):
        x = self.drop1(F.gelu(self.ln1(self.fc1(z))))
        return self.fc2(x)


class EEGDecoder(nn.Module):
    def __init__(self, latent_dim=32, output_dim=420, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, z):
        x = self.drop1(F.gelu(self.ln1(self.fc1(z))))
        return self.fc2(x)


class Classifier(nn.Module):
    def __init__(self, input_dim=32, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, z):
        x = self.drop1(F.gelu(self.ln1(self.fc1(z))))
        logits = self.fc2(x)
        return logits


class GazeMLP(nn.Module):
    def __init__(self, input_dim=9, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 64)
        self.ln2 = nn.LayerNorm(64)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, batch):
        x = batch["gaze"]
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        logits = self.fc3(x)
        return {"logits": logits}


class EEGMLP(nn.Module):
    def __init__(self, input_dim=420, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
    
    def forward(self, batch):
        x = batch["eeg"]
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        logits = self.fc3(x)
        return {"logits": logits}


class ConcatMLP(nn.Module):
    def __init__(self, eeg_dim=420, gaze_dim=9, num_classes=2, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(eeg_dim + gaze_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm(256)
        self.drop2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, batch):
        x = torch.cat([batch["eeg"], batch["gaze"]], dim=1)
        x = self.drop1(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop2(F.gelu(self.ln2(self.fc2(x))))
        logits = self.fc3(x)
        return {"logits": logits}


class EchoReadV0(nn.Module):
    def __init__(self, hidden_dim=32, dropout_rate=0.1):
        super().__init__()
        self.eeg_observer = EEGObserver(input_dim=420, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.gaze_observer = GazeObserver(input_dim=9, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.common_cause = CommonCauseEstimator(input_dim=hidden_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        
        self.gaze_decoder = GazeDecoder(latent_dim=hidden_dim, output_dim=9, dropout_rate=dropout_rate)
        self.eeg_decoder = EEGDecoder(latent_dim=hidden_dim, output_dim=420, dropout_rate=dropout_rate)
        
        self.eeg_to_gaze = GazeDecoder(latent_dim=hidden_dim, output_dim=9, dropout_rate=dropout_rate)
        self.gaze_to_eeg = EEGDecoder(latent_dim=hidden_dim, output_dim=420, dropout_rate=dropout_rate)
        
        self.classifier = Classifier(input_dim=hidden_dim, num_classes=2, dropout_rate=dropout_rate)
    
    def forward(self, batch):
        eeg = batch["eeg"]
        gaze = batch["gaze"]
        
        z_e = self.eeg_observer(eeg)
        z_g = self.gaze_observer(gaze)
        z_c = self.common_cause(z_e, z_g)
        
        gaze_hat_from_zc = self.gaze_decoder(z_c)
        eeg_hat_from_zc = self.eeg_decoder(z_c)
        
        gaze_hat_from_e = self.eeg_to_gaze(z_e)
        eeg_hat_from_g = self.gaze_to_eeg(z_g)
        
        logits = self.classifier(z_c)
        
        return {
            "logits": logits,
            "z_e": z_e,
            "z_g": z_g,
            "z_c": z_c,
            "gaze_hat_from_zc": gaze_hat_from_zc,
            "eeg_hat_from_zc": eeg_hat_from_zc,
            "gaze_hat_from_e": gaze_hat_from_e,
            "eeg_hat_from_g": eeg_hat_from_g
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model):
    summary = []
    summary.append(f"Model: {model.__class__.__name__}")
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append("-" * 50)
    
    if hasattr(model, "eeg_observer"):
        summary.append(f"EEGObserver: {count_parameters(model.eeg_observer):,} params")
    if hasattr(model, "gaze_observer"):
        summary.append(f"GazeObserver: {count_parameters(model.gaze_observer):,} params")
    if hasattr(model, "common_cause"):
        summary.append(f"CommonCauseEstimator: {count_parameters(model.common_cause):,} params")
    if hasattr(model, "gaze_decoder"):
        summary.append(f"GazeDecoder: {count_parameters(model.gaze_decoder):,} params")
    if hasattr(model, "eeg_decoder"):
        summary.append(f"EEGDecoder: {count_parameters(model.eeg_decoder):,} params")
    if hasattr(model, "eeg_to_gaze"):
        summary.append(f"EEG→Gaze Cross Decoder: {count_parameters(model.eeg_to_gaze):,} params")
    if hasattr(model, "gaze_to_eeg"):
        summary.append(f"Gaze→EEG Cross Decoder: {count_parameters(model.gaze_to_eeg):,} params")
    if hasattr(model, "classifier"):
        summary.append(f"Classifier: {count_parameters(model.classifier):,} params")
    
    return "\n".join(summary)