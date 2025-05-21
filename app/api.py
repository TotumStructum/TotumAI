from fastapi import FastAPI, Request
import torch
from model import TotumSeq2Seq
import torch.nn.functional as F
import uvicorn
import json

# === Завантаження словників і моделі ===
word2idx = torch.load("app/word2idx.pth")
idx2word = torch.load("app/idx2word.pth")

vocab_size = len(word2idx)
model = TotumSeq2Seq(vocab_size)
model.load_state_dict(torch.load("app/model_weights.pth"))
model.eval()

# === Функції ===
def tokenize(text):
    return [word2idx.get(w, word2idx["<UNK>"]) for w in text.lower().split()]

def detokenize(indices):
    return " ".join([idx2word.get(i, "<UNK>") for i in indices])

def generate_reply(input_text):
    input_ids = tokenize(input_text)
    input_tensor = torch.tensor([input_ids])
    _, (hidden, cell) = model.encoder(model.embedding(input_tensor))

    # Починаємо з <SOS>
    current = torch.tensor([[word2idx["<SOS>"]]])
    generated = []

    for _ in range(20):  # максимум 20 слів
        embedded = model.embedding(current)
        output, (hidden, cell) = model.decoder(embedded, (hidden, cell))
        logits = model.fc(output[:, -1, :])
        predicted_id = torch.argmax(F.softmax(logits, dim=-1), dim=-1).item()

        if predicted_id == word2idx["<EOS>"]:
            break

        generated.append(predicted_id)
        current = torch.tensor([[predicted_id]])

    return detokenize(generated)

# === FastAPI ===
app = FastAPI()

@app.post("/chat/{user_id}")
async def chat(user_id: str, request: Request):
    data = await request.json()
    user_input = data["message"]

    reply = generate_reply(user_input)
    return {"reply": reply}

# === Запуск (якщо запускати напряму) ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7300)
