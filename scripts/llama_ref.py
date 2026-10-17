#!/usr/bin/env python3
"""Эталон для проверки движка: собирает маленькую модель llama-архитектуры,
пишет её в формате GGUF и считает форвард независимой реализацией на numpy.

Зачем «эталон»: движок на C проверяется тем, что его логиты совпадают с
числами из этого файла до последнего знака. Одна и та же модель в том же файле
GGUF используется и для проверки формата.

Модель специально маленькая, но со всеми деталями настоящей:
RMSNorm, RoPE, grouped-query attention, SwiGLU, KV-кэш, BPE-токенизатор.

    python3 scripts/llama_ref.py out.gguf --tokens 8
"""
import argparse
import json
import math
import struct
import zlib

# ---------------------------------------------------------------- формат GGUF

GGUF_MAGIC = 0x46554747          # 'GGUF' в little endian
GGUF_TYPE_UINT8, GGUF_TYPE_INT8 = 0, 1
GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32 = 4, 5
GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL = 6, 7
GGUF_TYPE_STRING, GGUF_TYPE_ARRAY = 8, 9
GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 = 10, 11, 12

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

VERSION = 3          # в третьей версии длины строк и массивов — uint32
HEADER = struct.Struct('<IIQQ')


# Длины строк и массивов — ВСЕГДА uint64, и в v2, и в v3. В третьей версии
# изменился не размер поля длины, а кое-что другое; настоящие файлы от
# llama.cpp (Qwen2.5 и другие) пишут длину ключа восемью байтами.
#
# Раньше здесь стояло «в v3 — uint32». Из-за этого эталонная модель писалась
# в формате, который не читает ни наш движок, ни llama.cpp: разбор уезжал на
# четыре байта и упирался в «строка длиннее файла». Модель была нечитаемой,
# и сценарии «разложение по узлам» и «сверка конвейера» падали на файле,
# который сама же генератор и создавал.


def w_str(s):
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b


def w_alen(n):
    return struct.pack('<Q', n)


def w_value(vtype, val):
    if vtype == GGUF_TYPE_UINT32:
        return struct.pack('<I', val)
    if vtype == GGUF_TYPE_INT32:
        return struct.pack('<i', val)
    if vtype == GGUF_TYPE_UINT64:
        return struct.pack('<Q', val)
    if vtype == GGUF_TYPE_FLOAT32:
        return struct.pack('<f', val)
    if vtype == GGUF_TYPE_BOOL:
        return struct.pack('<B', 1 if val else 0)
    if vtype == GGUF_TYPE_STRING:
        return w_str(val)
    raise ValueError('тип %d в писателе не поддержан' % vtype)


def w_kv(key, vtype, val):
    """Ключ + значение. Массивы строк и строки пишем отдельно."""
    out = w_str(key) + struct.pack('<I', vtype)
    if vtype == GGUF_TYPE_ARRAY:
        # допустимы только два вида массивов: строки (tokens, merges) и uint32
        if isinstance(val, (list, tuple)) and val and isinstance(val[0], str):
            out += struct.pack('<I', GGUF_TYPE_STRING) + w_alen(len(val))
            for s in val:
                out += w_str(s)
        else:
            out += struct.pack('<I', GGUF_TYPE_UINT32) + w_alen(len(val))
            for x in val:
                out += struct.pack('<I', x)
    else:
        out += w_value(vtype, val)
    return out


def write_gguf(path, meta, tensors):
    """tensors: [(name, dims, float32 list)] — веса в том порядке, в каком идут."""
    # заголовок требует числа тензоров и пар метаданных до их записи
    head = HEADER.pack(GGUF_MAGIC, VERSION, len(tensors), len(meta))
    body = b''
    for name, vtype, val in meta:
        body += w_kv(name, vtype, val)

    info = b''
    offset = 0
    blob = b''
    for name, dims, data in tensors:
        info += w_str(name) + struct.pack('<I', len(dims))
        for d in dims:
            info += struct.pack('<Q', d)
        info += struct.pack('<I', GGML_TYPE_F32) + struct.pack('<Q', offset)
        blob += struct.pack('<%df' % len(data), *data)
        offset += len(data) * 4
        while offset % 32:                      # выравнивание general.alignment
            blob += b'\0'
            offset += 1

    # начало блока данных выравнивается, как это делает llama.cpp: читатель
    # вправе предполагать выравнивание, и без него все веса поедут
    head_len = len(head) + len(body) + len(info)
    pad = (-head_len) % 32
    with open(path, 'wb') as f:
        f.write(head + body + info + b'\0' * pad + blob)
    return path


# ------------------------------------------------------------- модель и словарь

def det_u32(*args):
    """Детерминированные псевдослучайные веса: модель должна воспроизводиться
    байт в байт, иначе сверка «распределённо против целиком» ничего не значит."""
    h = 2166136261
    for a in args:
        n = zlib.crc32(str(a).encode()) & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            h = ((h ^ (((n >> shift) & 0xFF) + 1)) * 16777619) & 0xFFFFFFFF
    return h


class Model:
    def __init__(self, n_layer=4, n_embd=64, n_head=4, n_head_kv=2, n_ff=128,
                 n_vocab=256, rope_base=10000.0, rms_eps=1e-5):
        self.n_layer, self.n_embd = n_layer, n_embd
        self.n_head, self.n_head_kv = n_head, n_head_kv
        self.head_dim = n_embd // n_head
        self.n_ff, self.n_vocab = n_ff, n_vocab
        self.rope_base, self.rms_eps = rope_base, rms_eps
        self.tensors = {}

    def _rand(self, name, *shape, scale=0.08):
        n = 1
        for s in shape:
            n *= s
        # вес зависит от имени тензора и от номера элемента: модель
        # воспроизводится байт в байт от запуска к запуску
        vals = [((det_u32(name, i) >> 8) / float(1 << 24) - 0.5) * scale for i in range(n)]
        self.tensors[name] = (list(shape), vals)
        return vals

    def build(self):
        ne = self.n_embd
        self._rand('token_embd.weight', ne, self.n_vocab)
        for i in range(self.n_layer):
            self._rand('blk.%d.attn_norm.weight' % i, ne, scale=1.0)
            self._rand('blk.%d.attn_q.weight' % i, ne, self.n_head * self.head_dim)
            self._rand('blk.%d.attn_k.weight' % i, ne, self.n_head_kv * self.head_dim)
            self._rand('blk.%d.attn_v.weight' % i, ne, self.n_head_kv * self.head_dim)
            self._rand('blk.%d.attn_output.weight' % i, self.n_head * self.head_dim, ne)
            self._rand('blk.%d.ffn_norm.weight' % i, ne, scale=1.0)
            self._rand('blk.%d.ffn_gate.weight' % i, ne, self.n_ff)
            self._rand('blk.%d.ffn_up.weight' % i, ne, self.n_ff)
            self._rand('blk.%d.ffn_down.weight' % i, self.n_ff, ne)
        self._rand('output_norm.weight', ne, scale=1.0)
        self._rand('output.weight', ne, self.n_vocab)
        return self


def build_vocab(n_vocab):
    """Байтовый BPE: 256 одиночных байтов + «слитные» токены."""
    # printable ASCII как есть, служебные байты — в виде <0xNN>
    toks = []
    for b in range(256):
        toks.append(chr(b) if 33 <= b < 127 else '<0x%02X>' % b)
    merges = []
    for i in range(n_vocab - 256):
        a = (det_u32('v', i) % 256)
        b = (det_u32('w', i) % 256)
        toks.append(toks[a] + toks[b])
        merges.append(toks[a] + ' ' + toks[b])
    return toks, merges


def write_model(path, m):
    toks, merges = build_vocab(m.n_vocab)
    meta = [
        ('general.architecture', GGUF_TYPE_STRING, 'llama'),
        ('llama.block_count', GGUF_TYPE_UINT32, m.n_layer),
        ('llama.context_length', GGUF_TYPE_UINT32, 256),
        ('llama.embedding_length', GGUF_TYPE_UINT32, m.n_embd),
        ('llama.feed_forward_length', GGUF_TYPE_UINT32, m.n_ff),
        ('llama.attention.head_count', GGUF_TYPE_UINT32, m.n_head),
        ('llama.attention.head_count_kv', GGUF_TYPE_UINT32, m.n_head_kv),
        ('llama.attention.layer_norm_rms_epsilon', GGUF_TYPE_FLOAT32, m.rms_eps),
        ('llama.rope.freq_base', GGUF_TYPE_FLOAT32, m.rope_base),
        ('llama.rope.dimension_count', GGUF_TYPE_UINT32, m.head_dim),
        ('tokenizer.ggml.model', GGUF_TYPE_STRING, 'gpt2'),
        ('tokenizer.ggml.tokens', GGUF_TYPE_ARRAY, toks),
        ('tokenizer.ggml.merges', GGUF_TYPE_ARRAY, merges),
        ('tokenizer.ggml.bos_token_id', GGUF_TYPE_UINT32, 1),
        ('tokenizer.ggml.eos_token_id', GGUF_TYPE_UINT32, 2),
        ('tokenizer.ggml.add_bos_token', GGUF_TYPE_BOOL, True),
        ('tokenizer.ggml.add_eos_token', GGUF_TYPE_BOOL, False),
    ]
    tensors = [(name, shape, data) for name, (shape, data) in m.tensors.items()]
    return write_gguf(path, meta, tensors)


# ------------------------------------------------------------------ numpy

def rms_norm(x, w, eps):
    """RMSNorm по столбцам: x имеет вид [ne, T], w — [ne]."""
    import numpy as np
    ss = (x * x).sum(axis=0) / x.shape[0]
    return x / np.sqrt(ss + eps) * w[:, None]


def forward(m, tokens):
    """Форвард всей последовательности с KV-кэшем; возвращает логиты
    последнего токена. Определения операций — как в llama.cpp:
    y = W @ x, где W лежит как [ne0, ne1], элемент (i, j) по адресу i*ne1 + j."""
    import numpy as np
    ne, hd, nh, nkv = m.n_embd, m.head_dim, m.n_head, m.n_head_kv
    n_rep = nh // nkv

    def W(name):
        """Матрица веса в виде, в каком ggml их читает: [ne1, ne0].

        В файле элемент (i0, i1) стоит по адресу i0 + i1*ne0, то есть
        непрерывное измерение ne0 — это вход. Поэтому плоский буфер читается
        как [ne1, ne0], и y = W @ x уже есть matvec, без транспонирования.

        Раньше здесь стояло reshape(shape), то есть [ne0, ne1] с шагом ne1, и
        дальше каждое умножение компенсировалось .T. Для квадратных весов это
        выглядело верно, а для неквадратных (эмбеддинги, ffn) — нет: строка
        бралась не оттуда, и модель получала вход из чужих элементов.
        """
        shape, data = m.tensors[name]
        a = np.array(data, dtype=np.float32)
        if len(shape) == 1:
            return a                     # вектор (нормы): транспонировать нечего
        return a.reshape(shape[1], shape[0])

    def rope(vec, pos):
        out = vec.copy()
        half = hd // 2
        freqs = 1.0 / (m.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / hd))
        ang = pos * freqs
        c, s = np.cos(ang), np.sin(ang)
        out[:half] = vec[:half] * c - vec[half:half * 2] * s
        out[half:half * 2] = vec[half:half * 2] * c + vec[:half] * s
        return out

    # Токены идут в эмбеддингах вплотную блоками по n_embd: строка t стоит по
    # адресу t*ne. W отдаёт [n_vocab, ne], так что нужна строка, а не столбец.
    x = W('token_embd.weight')[tokens].T             # [ne, T]
    T = x.shape[1]

    for i in range(m.n_layer):
        h = rms_norm(x, W('blk.%d.attn_norm.weight' % i), m.rms_eps)
        q = (W('blk.%d.attn_q.weight' % i) @ h).reshape(nh, hd, T)
        k = (W('blk.%d.attn_k.weight' % i) @ h).reshape(nkv, hd, T)
        v = (W('blk.%d.attn_v.weight' % i) @ h).reshape(nkv, hd, T)

        out = np.zeros((nh, hd, T), dtype=np.float32)
        for g in range(nh):
            kvg = g // n_rep
            for t in range(T):
                qt = rope(q[g, :, t], t)
                kt = rope(k[kvg, :, t], t)
                vt = v[kvg, :, t]
                scores = []
                for t2 in range(t + 1):
                    k2 = kt if t2 == t else rope(k[kvg, :, t2], t2)
                    scores.append(float((qt * k2).sum()) / math.sqrt(hd))
                mx = max(scores)
                ex = [math.exp(s - mx) for s in scores]
                tot = sum(ex)
                acc = np.zeros(hd, dtype=np.float32)
                for t2 in range(t + 1):
                    v2 = vt if t2 == t else v[kvg, :, t2]
                    acc += (ex[t2] / tot) * v2
                out[g, :, t] = acc

        merged = out.reshape(nh * hd, T)
        x = x + W('blk.%d.attn_output.weight' % i) @ merged

        h = rms_norm(x, W('blk.%d.ffn_norm.weight' % i), m.rms_eps)
        gate = W('blk.%d.ffn_gate.weight' % i) @ h
        up = W('blk.%d.ffn_up.weight' % i) @ h
        act = (gate / (1.0 + np.exp(-gate))) * up                 # SwiGLU
        x = x + W('blk.%d.ffn_down.weight' % i) @ act

    xf = rms_norm(x[:, -1:], W('output_norm.weight'), m.rms_eps)[:, 0]
    return W('output.weight') @ xf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--tokens', type=int, default=8)
    ap.add_argument('--prompt', default='distcomp')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    m = Model().build()
    write_model(args.out, m)

    import numpy as np
    toks, _ = build_vocab(m.n_vocab)
    ids = [1] + [toks.index(c) if c in toks else 0 for c in args.prompt]
    ids = [i for i in ids if i < m.n_vocab]

    logits = forward(m, ids)
    greedy = int(np.argmax(logits))
    top = np.argsort(-logits)[:5]
    data = {
        'file': args.out,
        'shape': {'n_layer': m.n_layer, 'n_embd': m.n_embd, 'n_head': m.n_head,
                  'n_head_kv': m.n_head_kv, 'n_ff': m.n_ff, 'n_vocab': m.n_vocab},
        'prompt_ids': ids,
        'logits': [float(v) for v in logits[:16]],
        'greedy': greedy,
        'top5': [int(i) for i in top],
    }
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(data, f)
    print('модель %s: слоёв %d, embd %d, голов %d/%d, ff %d, vocab %d' %
          (args.out, m.n_layer, m.n_embd, m.n_head, m.n_head_kv, m.n_ff, m.n_vocab))
    print('токены промпта: %s' % ids)
    print('логиты (первые 8): %s' % ['%.6f' % v for v in logits[:8]])
    print('жадный выбор: %d (%.6f)' % (greedy, float(logits[greedy])))
    return data


if __name__ == '__main__':
    main()