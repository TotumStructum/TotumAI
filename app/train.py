import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import TotumSeq2Seq

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data/dialogues.txt")

# === 1. Завантаження і парсинг даних ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

pairs = []
for i in range(0, len(lines), 2):
    input_line = lines[i].replace("User: ", "")
    target_line = lines[i+1].replace("Bot: ", "")
    pairs.append((input_line, target_line))

# === 2. Словник ===
all_text = " ".join([inp + " " + out for inp, out in pairs])
tokens = sorted(set(all_text.lower().split()))
tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"] + tokens
word2idx = {word: i for i, word in enumerate(tokens)}
idx2word = {i: word for word, i in word2idx.items()}

def tokenize(text):
    return [word2idx.get(word, word2idx["<UNK>"]) for word in text.lower().split()]

def pad(seq, max_len):
    return seq + [word2idx["<PAD>"]] * (max_len - len(seq))

# === 3. Обчислюємо max довжини один раз ===
max_inp = max(len(tokenize(inp)) for inp, _ in pairs)
max_out = max(len(tokenize(out)) for _, out in pairs) + 2  # <SOS> + <EOS>

# === 4. Формуємо тензори ===
inputs, targets = [], []
for inp, out in pairs:
    inp_ids = tokenize(inp)
    out_ids = [word2idx["<SOS>"]] + tokenize(out) + [word2idx["<EOS>"]]

    inputs.append(pad(inp_ids, max_inp))
    targets.append(pad(out_ids, max_out))

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# === 5. Створюємо модель ===
model = TotumSeq2Seq(vocab_size=len(word2idx))
loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === 6. Навчання ===
EPOCHS = 100

for epoch in range(1, EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    output = model(inputs, targets[:, :-1])
    loss = loss_fn(output.reshape(-1, output.shape[-1]), targets[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {loss.item():.4f}")

# === 7. Збереження ===
torch.save(model.state_dict(), "app/model_weights.pth")
torch.save(word2idx, "app/word2idx.pth")
torch.save(idx2word, "app/idx2word.pth")

print("✅ Навчання завершено успішно")