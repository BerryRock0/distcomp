#!/usr/bin/env python3
"""Выгружает настоящие блоки каждого K-кванта из модели + эталон ggml.

Берёт блоки прямо из весов Qwen2.5-Q2_K (там смесь Q2_K/Q3_K/Q4_K/Q6_K) —
это те данные, на которых движок реально считает. Для каждого блока пишутся
сырые байты и все 256 значений, полученных пакетом gguf.

    python3 dump_blocks.py model.gguf out.bin
"""
import sys
import numpy as np
from gguf.quants import dequantize, GGMLQuantizationType

BSZ = {10: 84, 11: 110, 12: 144, 13: 176, 14: 210, 15: 292}
NAME = {10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K'}


def main():
    model, out = sys.argv[1], sys.argv[2]
    from gguf import GGUFReader
    r = GGUFReader(model)
    per_type = {}
    for t in r.tensors:
        qt = t.tensor_type.value
        if qt not in BSZ:
            continue
        data = np.frombuffer(t.data, dtype=np.uint8)
        nb = data.size // BSZ[qt]
        # по несколько блоков с разных тензоров
        cur = per_type.setdefault(qt, [])
        if len(cur) >= 24 or not nb:
            continue
        take = min(8, nb)
        idx = np.linspace(0, nb - 1, take).astype(int)
        for i in idx:
            cur.append(data[i * BSZ[qt]:(i + 1) * BSZ[qt]])

    with open(out, 'wb') as f:
        f.write(np.uint32(len(per_type)).tobytes())
        for qt, blocks in sorted(per_type.items()):
            blocks = np.array(blocks, dtype=np.uint8)
            truth = np.concatenate([dequantize(b, np.uint8(qt)) for b in blocks])
            # блоки идут вплотную, реальным размером: gguf_dequant сам шагает
            # по ним размером блока, и дополнять их нельзя
            f.write(np.uint32(qt).tobytes())
            f.write(np.uint32(len(blocks)).tobytes())
            f.write(blocks.tobytes())
            f.write(truth.astype(np.float32).tobytes())
            print('%-6s блоков %4d' % (NAME.get(qt, qt), len(blocks)), file=sys.stderr)


if __name__ == '__main__':
    main()
