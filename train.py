
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# NCVES - Neural Cryptographic Visual Encoding System
# Research Proof of Concept
# ============================================================
class NCVESEncoder(nn.Module):
    def __init__(
        self,
        message_bits=128,
        key_bits=64,
        image_size=128
    ):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Sequential(
            nn.Linear(message_bits + key_bits, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, image_size * image_size),
            nn.Sigmoid()
        )

    def forward(self, message, key):
        x = torch.cat([message, key], dim=1)
        img = self.fc(x)
        img = img.view(
            -1,
            1,
            self.image_size,
            self.image_size
        )

        return img


class PhysicalChannel(nn.Module):
    def __init__(self):
        super().__init__()

    def gaussian_blur(self, x):
        kernel = torch.tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ], device=x.device)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, 3, 3)
        return F.conv2d(
            x,
            kernel,
            padding=1
        )

    def forward(self, x):
        x = self.gaussian_blur(x)
        noise = 0.05 * torch.randn_like(x)
        x = x + noise
        x = torch.clamp(x, 0, 1)
        return x


class NCVESDecoder(nn.Module):
    def __init__(
        self,
        message_bits=128
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(
            64,
            message_bits
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        msg = self.fc(feat)
        return msg


class NCVES(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = NCVESEncoder()
        self.channel = PhysicalChannel()
        self.decoder = NCVESDecoder()

    def forward(
        self,
        message,
        key
    ):
        encoded = self.encoder(
            message,
            key
        )
        degraded = self.channel(
            encoded
        )
        decoded = self.decoder(
            degraded
        )
        return (
            encoded,
            degraded,
            decoded
        )

def reconstruction_loss(
    pred,
    target
):
    return nn.BCEWithLogitsLoss()(
        pred,
        target
    )

def key_separation_loss(
    model,
    message,
    wrong_key
):
    _, _, decoded = model(
        message,
        wrong_key
    )
    return -nn.BCEWithLogitsLoss()(
        decoded,
        message
    )


def total_loss(
    model,
    message,
    correct_key,
    wrong_key,
    lambda_key=0.1
):
    _, _, decoded_good = model(
        message,
        correct_key
    )
    rec = reconstruction_loss(
        decoded_good,
        message
    )
    sep = key_separation_loss(
        model,
        message,
        wrong_key
    )

    return rec + lambda_key * sep


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = NCVES().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )
    epochs = 1000
    batch_size = 32
    for step in range(epochs):
        message = torch.randint(
            0,
            2,
            (batch_size, 128),
            device=device
        ).float()
        key = torch.randint(
            0,
            2,
            (batch_size, 64),
            device=device
        ).float()
        wrong_key = torch.randint(
            0,
            2,
            (batch_size, 64),
            device=device
        ).float()
        loss = total_loss(
            model,
            message,
            key,
            wrong_key
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(
                f"step={step:05d} loss={loss.item():.6f}"
            )

    print("Training finished.")
    with torch.no_grad():
        test_msg = torch.randint(
            0,
            2,
            (1, 128),
            device=device
        ).float()
        test_key = torch.randint(
            0,
            2,
            (1, 64),
            device=device
        ).float()
        encoded, degraded, decoded = model(
            test_msg,
            test_key
        )

        print("Encoded image shape:", encoded.shape)
        print("Decoded logits shape:", decoded.shape)


if __name__ == "__main__":
    main()
