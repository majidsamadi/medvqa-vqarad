import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel

class FusionTransformer(nn.Module):
    def __init__(
        self,
        num_answers: int,
        text_model_name: str = "distilbert-base-uncased",
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        max_text_len: int = 32,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.max_text_len = max_text_len

        self.text = DistilBertModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text.config.hidden_size, d_model)

        cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(cnn.children())[:-2])

        if freeze_backbones:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.text.parameters():
                p.requires_grad = False

        self.img_proj = nn.Linear(512, d_model)

        self.type_emb = nn.Embedding(2, d_model)

        max_img_tokens = 7 * 7
        self.pos_emb = nn.Parameter(torch.zeros(1, max_text_len + max_img_tokens, d_model))

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)


        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_answers),
        )

    def forward(self, images, input_ids, attention_mask):
        bsz = input_ids.size(0)

        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        text_tok = self.text_proj(text_out)
        text_type = self.type_emb(torch.zeros((bsz, text_tok.size(1)), device=text_tok.device, dtype=torch.long))
        text_tok = text_tok + text_type

        feat = self.backbone(images)
        feat = feat.flatten(2).transpose(1, 2)
        img_tok = self.img_proj(feat)
        img_type = self.type_emb(torch.ones((bsz, img_tok.size(1)), device=img_tok.device, dtype=torch.long))
        img_tok = img_tok + img_type

        x = torch.cat([text_tok, img_tok], dim=1)
        x = x + self.pos_emb[:, : x.size(1), :]

        img_mask = torch.ones((bsz, img_tok.size(1)), device=attention_mask.device, dtype=attention_mask.dtype)
        attn = torch.cat([attention_mask, img_mask], dim=1)
        pad_mask = (attn == 0)

        out = self.encoder(x, src_key_padding_mask=pad_mask)
        cls = out[:, 0, :]
        return self.classifier(cls)
