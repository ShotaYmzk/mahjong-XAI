# train_template.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# ─────────────────────────────────────────────────────────────────────────────
#  1) デバイス設定
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
#  2) サンプル Dataset (適宜書き換え)
# ─────────────────────────────────────────────────────────────────────────────
class SampleSequenceDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=50, vocab_size=1000):
        super().__init__()
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, vocab_size, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]       # [seq_len]
        y = self.labels[idx]     # scalar
        return x, y

# ─────────────────────────────────────────────────────────────────────────────
#  3) Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)             # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1)     # [max_len, 1]
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len]

# ─────────────────────────────────────────────────────────────────────────────
#  4) Transformer モデル本体
# ─────────────────────────────────────────────────────────────────────────────
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, nhid=2048, nlayers=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, src):
        # src: [batch, seq_len]
        x = self.embedding(src) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)                    # [batch, seq_len, d_model]
        x = x.mean(dim=1)                                   # 平均プーリング
        return self.fc_out(x)                              # [batch, vocab_size]

# ─────────────────────────────────────────────────────────────────────────────
#  5) ハイパーパラメータ＆DataLoader
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 128
EPOCHS      = 10
LR          = 1e-4
VOCAB_SIZE  = 1000
SEQ_LEN     = 50

train_dataset = SampleSequenceDataset(num_samples=20000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                           num_workers=4, pin_memory=True if DEVICE.type=='cuda' else False)

# ─────────────────────────────────────────────────────────────────────────────
#  6) モデル／オプティマイザ／損失
# ─────────────────────────────────────────────────────────────────────────────
model     = TransformerModel(VOCAB_SIZE).to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# ─────────────────────────────────────────────────────────────────────────────
#  7) 学習ループ
# ─────────────────────────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_loader, 1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)                 # [batch, vocab_size]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Epoch[{epoch}/{EPOCHS}] Batch[{batch_idx}/{len(train_loader)}]  "
                  f"Loss: {running_loss/batch_idx:.4f}")

    elapsed = time.time() - start_time
    avg_loss = running_loss / len(train_loader)
    print(f"==> Epoch {epoch} completed in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
#  8) モデル保存
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), f"checkpoints/transformer_epoch{EPOCHS}.pth")
print("Model saved.")
