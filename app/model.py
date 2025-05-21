import torch
import torch.nn as nn

class TotumSeq2Seq(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=256):
        super(TotumSeq2Seq, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt):
        # src: [batch_size, src_len]
        # tgt: [batch_size, tgt_len]
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        embedded_tgt = self.embedding(tgt)
        output, _ = self.decoder(embedded_tgt, (hidden, cell))
        prediction = self.fc(output)
        return prediction
