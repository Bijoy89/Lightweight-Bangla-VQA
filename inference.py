import os, re, json, unicodedata
import torch
import timm
import torchvision.transforms as T
from PIL import Image
from model import VQAModel, DEVICE, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

BANGLA_PUNCT = re.compile(r'[।,!?;:\.\"\'\'\"\"\(\)\[\]\{\}]+')
MULTI_SPACE  = re.compile(r'\s+')

BANGLA_DIGIT_MAP = {
    '০':'শূন্য','১':'এক','২':'দুই','৩':'তিন',
    '৪':'চার', '৫':'পাঁচ','৬':'ছয়','৭':'সাত',
    '৮':'আট', '৯':'নয়'
}


def normalize_bangla(text):
    if not isinstance(text, str): return ''
    text = unicodedata.normalize('NFC', text.strip())
    text = BANGLA_PUNCT.sub(' ', text)
    text = MULTI_SPACE.sub(' ', text)
    return text.strip()


def word_tokenize(text):
    return normalize_bangla(text).split()


def encode_seq(tokens, stoi, max_len):
    ids  = [SOS_IDX]
    ids += [stoi.get(t, UNK_IDX) for t in tokens[:max_len]]
    ids += [EOS_IDX]
    ids += [PAD_IDX] * (max_len + 2 - len(ids))
    return torch.tensor(ids[:max_len + 2], dtype=torch.long)


def decode_seq(ids, itos):
    words = []
    for i in ids:
        if i == EOS_IDX: break
        if i not in (PAD_IDX, SOS_IDX):
            words.append(itos.get(i, '<UNK>'))
    return ' '.join(words)


eval_tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class BanglaVQAPipeline:
    def __init__(
        self,
        checkpoint_path='checkpoints/best_model.pt',
        q_vocab_path='vocab/q_stoi.json',
        a_vocab_path='vocab/a_stoi.json',
        hidden_dim=512,
        fusion_type='concat',
        emb_dim=300,
        max_q_len=20,
        max_a_len=10,
        beam_size=3,
    ):
        self.max_q_len = max_q_len
        self.max_a_len = max_a_len
        self.beam_size = beam_size

        # ── Load vocabs ──
        print('Loading vocabs...')
        with open(q_vocab_path, encoding='utf-8') as f:
            self.q_stoi = json.load(f)
        with open(a_vocab_path, encoding='utf-8') as f:
            self.a_stoi = json.load(f)
        self.a_itos = {int(i): w for w, i in self.a_stoi.items()}

        q_vocab_size = len(self.q_stoi)   # 4,750 with min_freq=2
        a_vocab_size = len(self.a_stoi)   # 1,747 with min_freq=2

        print(f'  Q vocab: {q_vocab_size:,} tokens')
        print(f'  A vocab: {a_vocab_size:,} tokens')

        # Load EfficientNet-B0
        print('Loading EfficientNet-B0...')
        self.effnet = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,
            global_pool=''
        ).to(DEVICE)
        for p in self.effnet.parameters():
            p.requires_grad = False
        self.effnet.eval()

        #  Build VQA model with CORRECT vocab sizes from JSON
        print('Loading VQA model...')
        print(f'  Config: fusion={fusion_type} | hidden={hidden_dim} | emb={emb_dim}')
        self.model = VQAModel(
            q_vocab_size=q_vocab_size,
            a_vocab_size=a_vocab_size,
            hidden_dim=hidden_dim,
            fusion_type=fusion_type,
            emb_dim=emb_dim,
        ).to(DEVICE)

        #  Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        print(f'Model ready!')
        print(f'  Epoch : {ckpt.get("epoch", "?")}')
        print(f'  Score : {ckpt.get("score", 0):.4f}')
        print(f'  Device: {DEVICE}')

    def predict(self, image: Image.Image, question: str) -> dict:
        # Process image
        img_tensor = eval_tfm(image.convert('RGB')).unsqueeze(0).to(DEVICE)

        # Process question
        tokens = word_tokenize(question)
        q_ids  = encode_seq(tokens, self.q_stoi, self.max_q_len
                            ).unsqueeze(0).to(DEVICE)

        # Extract features
        with torch.no_grad():
            feat_map = self.effnet(img_tensor)

        # Beam search
        pred_ids = self.model.beam_search(
            feat_map, q_ids,
            beam_size=self.beam_size,
            max_len=self.max_a_len
        )

        answer = decode_seq(pred_ids, self.a_itos)

        return {
            'answer':        answer if answer.strip() else 'উত্তর পাওয়া যায়নি',
            'question':      question,
            'tokens':        tokens,
            'answer_tokens': answer.split() if answer.strip() else [],
        }