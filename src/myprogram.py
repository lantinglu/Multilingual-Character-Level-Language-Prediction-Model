
import os
import re
import csv
import sys
import math
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser
from collections import Counter


class CharNgramModel:
    def __init__(self, max_order=5):
        self.max_order = max_order
        self.counts = {n: {} for n in range(1, max_order + 1)}
        self.unigram_counts = Counter()
        self.total_chars = 0

    def train(self, texts):
        total = len(texts)
        t0 = time.time()
        report_interval = max(1, total // 20)
        counts = self.counts
        max_order = self.max_order

        for idx, text in enumerate(texts):
            tlen = len(text)
            self.total_chars += tlen

            self.unigram_counts.update(text)

            for i in range(tlen):
                next_char = text[i]
                for n in range(1, min(max_order + 1, i + 2)):
                    ctx = text[i - n + 1:i] if n > 1 else ''
                    bucket = counts[n]
                    if ctx in bucket:
                        inner = bucket[ctx]
                        inner[next_char] = inner.get(next_char, 0) + 1
                    else:
                        bucket[ctx] = {next_char: 1}

            if (idx + 1) % report_interval == 0:
                elapsed = time.time() - t0
                pct = (idx + 1) / total * 100
                speed = (idx + 1) / elapsed
                eta = (total - idx - 1) / speed


    def predict_top_k(self, context_str, k=3):
        combined = Counter()
        weights = {n: 2.0 ** n for n in range(1, self.max_order + 1)}

        for n in range(1, self.max_order + 1):
            if len(context_str) < n - 1:
                continue
            ctx = context_str[-(n - 1):] if n > 1 else ''
            char_counts = self.counts[n].get(ctx)
            if char_counts:
                total = sum(char_counts.values())
                for ch, cnt in char_counts.items():
                    combined[ch] += weights[n] * (cnt / total)

        if self.total_chars > 0:
            for ch, cnt in self.unigram_counts.most_common(50):
                combined[ch] += 0.5 * (cnt / self.total_chars)

        if not combined:
            return list(' etaoinsrhld'[:k])

        top = combined.most_common(k)
        return [ch for ch, _ in top]

    def get_distribution(self, context_str):
        combined = Counter()
        weights = {n: 2.0 ** n for n in range(1, self.max_order + 1)}

        for n in range(1, self.max_order + 1):
            if len(context_str) < n - 1:
                continue
            ctx = context_str[-(n - 1):] if n > 1 else ''
            char_counts = self.counts[n].get(ctx)
            if char_counts:
                total = sum(char_counts.values())
                for ch, cnt in char_counts.items():
                    combined[ch] += weights[n] * (cnt / total)

        if self.total_chars > 0:
            for ch, cnt in self.unigram_counts.most_common(100):
                combined[ch] += 0.5 * (cnt / self.total_chars)

        return combined


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, src):
        B, T = src.shape
        device = src.device

        T = min(T, self.max_len)
        src = src[:, -T:]

        positions = torch.arange(T, device=device).unsqueeze(0)
        mask = torch.triu(torch.ones(T, T, device=device) * float('-inf'), diagonal=1)

        x = self.embedding(src) + self.pos_embedding(positions)
        output = self.transformer(x, mask=mask)
        output = self.ln_f(output)
        return self.fc_out(output)


class TextDataset(Dataset):
    def __init__(self, data, vocab, seq_len=256):
        self.seq_len = seq_len
        self.unk_id = vocab.get('<unk>', 2)
        self.sos_id = vocab.get('<sos>', 1)
        self.pad_id = vocab.get('<pad>', 0)

        self.samples = []
        for text in data:
            tokenized = [vocab.get(c, self.unk_id) for c in text]
            if len(tokenized) < 2:
                continue
            self.samples.append(tokenized)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokenized = self.samples[idx]

        if len(tokenized) > self.seq_len:
            start_idx = random.randint(0, len(tokenized) - self.seq_len)
            tokenized = tokenized[start_idx: start_idx + self.seq_len]

        input_seq = [self.sos_id] + tokenized[:-1]
        target_seq = tokenized

        pad_len = self.seq_len - len(input_seq)
        if pad_len > 0:
            input_seq += [self.pad_id] * pad_len
            target_seq += [self.pad_id] * pad_len

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def clean_text(text):
    """Remove <sos>, <eos> tags and clean up text."""
    text = re.sub(r'</?sos>', '', text)
    text = re.sub(r'</?eos>', '', text)
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1].strip()
    return text


def load_csv_data(path):
    data = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline().strip()
            f.seek(0)

            if ',' in first_line and first_line.lower().startswith('lang'):
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get('content', '').strip()
                    content = clean_text(content)
                    if content and len(content) > 1:
                        data.append(content)
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.isdigit():
                        continue
                    content = clean_text(line)
                    if content and len(content) > 1:
                        data.append(content)
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return data

SPECIAL_TOKENS = {'<pad>', '<sos>', '<unk>'}
MODEL_CONFIG = {
    'd_model': 384,
    'nhead': 6,
    'num_layers': 6,
    'max_len': 512,
    'dropout': 0.1,
}


class MyModel:
    def __init__(self, vocab=None):
        self.vocab = vocab
        self.model = None
        self.ngram = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.vocab:
            self.id2char = {v: k for k, v in self.vocab.items()}
            self.special_ids = {v for k, v in self.vocab.items() if k in SPECIAL_TOKENS}
        else:
            self.id2char = {}
            self.special_ids = set()

    @classmethod
    def load_training_data(cls):
        data = []
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        data_dir = os.path.join(project_root, 'data')
        if not os.path.isdir(data_dir):
            return data

        for filename in sorted(os.listdir(data_dir)):
            if filename.lower().endswith(('.txt', '.csv')):
                path = os.path.join(data_dir, filename)
                file_data = load_csv_data(path)
                print(f"  Loaded {len(file_data)} lines from {filename}")
                data.extend(file_data)

        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        if os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    data.append(line.rstrip('\n'))
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        if not data:
            print("No valid data loaded.")
            sys.exit(1)

        print(f"Loaded {len(data)} lines of data. ")
        t_start = time.time()

        counter = Counter()
        for i, line in enumerate(data):
            counter.update(line)
            if (i + 1) % 2000000 == 0:
                print(f"scan: {i+1}/{len(data)} line...")
        chars = [c for c, count in counter.items() if count >= 2]

        self.vocab = {'<pad>': 0, '<sos>': 1, '<unk>': 2}
        for i, c in enumerate(sorted(chars)):
            self.vocab[c] = i + 3
        self.id2char = {v: k for k, v in self.vocab.items()}
        self.special_ids = {v for k, v in self.vocab.items() if k in SPECIAL_TOKENS}
        print(f"Size: {len(self.vocab)}")

        self.ngram = CharNgramModel(max_order=5)
        self.ngram.train(data)

        vocab_size = len(self.vocab)
        self.model = CharTransformer(
            vocab_size,
            d_model=MODEL_CONFIG['d_model'],
            nhead=MODEL_CONFIG['nhead'],
            num_layers=MODEL_CONFIG['num_layers'],
            max_len=MODEL_CONFIG['max_len'],
            dropout=MODEL_CONFIG['dropout'],
        ).to(self.device)


        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.98))
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        dataset = TextDataset(data, self.vocab, seq_len=256)
        dataloader = DataLoader(
            dataset, batch_size=128, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True,
        )

        num_epochs = 10
        total_steps = num_epochs * len(dataloader)
        warmup_steps = min(2000, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_loss = float('inf')
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            count = 0

            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                total_loss += loss.item()
                count += 1


            avg_loss = total_loss / count

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save(work_dir)


    def run_pred(self, data):
        preds = []
        has_transformer = self.model is not None
        has_ngram = self.ngram is not None

        if has_transformer:
            self.model.eval()

        unk_id = self.vocab.get('<unk>', 2) if self.vocab else 2
        sos_id = self.vocab.get('<sos>', 1) if self.vocab else 1
        window_size = 256

        alpha = 0.7 if has_transformer else 0.0

        for inp in data:
            try:
                top3 = self._predict_single(
                    inp, has_transformer, has_ngram,
                    unk_id, sos_id, window_size, alpha
                )
                preds.append("".join(top3))
            except Exception:
                preds.append(' et')

        return preds

    def _predict_single(self, inp, has_transformer, has_ngram, unk_id, sos_id, window_size, alpha):
        combined = Counter()

        if has_transformer and self.vocab:
            context = inp[-window_size:]
            input_ids = [self.vocab.get(c, unk_id) for c in context]
            input_tensor = torch.tensor([[sos_id] + input_ids], device=self.device)

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[0, -1, :]

                for sid in self.special_ids:
                    last_logits[sid] = float('-inf')

                probs = torch.softmax(last_logits, dim=-1)
                top_k = min(20, probs.shape[0])
                top_probs, top_indices = torch.topk(probs, top_k)

                for p, idx in zip(top_probs, top_indices):
                    char = self.id2char.get(idx.item())
                    if char and char not in SPECIAL_TOKENS:
                        combined[char] += alpha * p.item()

        if has_ngram:
            ngram_dist = self.ngram.get_distribution(inp)
            if ngram_dist:
                total = sum(ngram_dist.values())
                for ch, score in ngram_dist.most_common(20):
                    combined[ch] += (1.0 - alpha) * (score / total)

        if not combined:
            return [' ', 'e', 't']

        top3 = [ch for ch, _ in combined.most_common(3)]

        fallback_chars = [' ', 'e', 't', 'a', 'o', 'i', 'n', 's']
        while len(top3) < 3:
            for fc in fallback_chars:
                if fc not in top3:
                    top3.append(fc)
                    break
            else:
                top3.append(' ')

        return top3[:3]

    def save(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)

        if self.model is not None:
            state = {
                'model_state': self.model.state_dict(),
                'vocab': self.vocab,
                'config': MODEL_CONFIG,
            }
            torch.save(state, os.path.join(work_dir, 'model.pth'))
            print("Transformer model saved.")

        if self.ngram is not None:
            with open(os.path.join(work_dir, 'ngram.pkl'), 'wb') as f:
                pickle.dump(self.ngram, f)
            print("N-gram model saved.")

    @classmethod
    def load(cls, work_dir):
        obj = cls()

        model_path = os.path.join(work_dir, 'model.pth')
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location='cpu')
            obj.vocab = state['vocab']
            obj.id2char = {v: k for k, v in obj.vocab.items()}
            obj.special_ids = {v for k, v in obj.vocab.items() if k in SPECIAL_TOKENS}

            config = state.get('config', MODEL_CONFIG)
            obj.model = CharTransformer(
                len(obj.vocab),
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                max_len=config.get('max_len', 512),
                dropout=config.get('dropout', 0.1),
            )
            obj.model.load_state_dict(state['model_state'])
            obj.model.to(obj.device)
            obj.model.eval()

        ngram_path = os.path.join(work_dir, 'ngram.pkl')
        if os.path.exists(ngram_path):
            with open(ngram_path, 'rb') as f:
                obj.ngram = pickle.load(f)

        if obj.model is None and obj.ngram is None:
            print("No models found! ")

        return obj


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('mode', choices=('train', 'test'))
    parser.add_argument('--work_dir', default='work')
    parser.add_argument('--test_data', default='example/input.txt')
    parser.add_argument('--test_output', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    if not os.path.isabs(args.work_dir):
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        args.work_dir = os.path.join(project_root, args.work_dir)

    if args.mode == 'train':
        os.makedirs(args.work_dir, exist_ok=True)
        model = MyModel()
        train_data = MyModel.load_training_data()
        model.run_train(train_data, args.work_dir)
        model.save(args.work_dir)
    elif args.mode == 'test':
        model = MyModel.load(args.work_dir)
        test_data = MyModel.load_test_data(args.test_data)
        pred = model.run_pred(test_data)
        model.write_pred(pred, args.test_output)