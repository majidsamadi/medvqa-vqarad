import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import resnet18, ResNet18_Weights

class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size: int, num_answers: int, emb_dim: int = 256, lstm_hidden: int = 256, freeze_cnn: bool = True):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, lstm_hidden, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(512 + 2 * lstm_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

    def forward(self, images, input_ids, attention_mask):
        img_vec = self.cnn(images)

        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        x = self.embedding(input_ids)

        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        idx = (lengths - 1).to(out.device).view(-1, 1, 1)
        last = out.gather(1, idx.expand(-1, 1, out.size(2))).squeeze(1)

        fused = torch.cat([img_vec, last], dim=1)
        return self.classifier(fused)
