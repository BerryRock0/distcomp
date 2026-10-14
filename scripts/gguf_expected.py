#!/usr/bin/env python3
"""Печатает ожидаемые жадные токены для промпта — эталон на numpy.

Служит внешним оракулом для распределённого конвейера: `dcctl llm prompt`
сверяет конвейер со СВОИМ локальным прогоном (тем же движком на C), а этот
скрипт считает ту же последовательность независимой реализацией на numpy.

    scripts/gguf_expected.py --prompt distcomp --tokens 4
    # -> 171 16 66 120
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt', default='distcomp')
    ap.add_argument('--tokens', type=int, default=8)
    args = ap.parse_args()

    import numpy as np
    import llama_ref as R

    m = R.Model().build()
    toks, _ = R.build_vocab(m.n_vocab)
    ids = [1] + [toks.index(c) if c in toks else 0 for c in args.prompt]
    ids = [i for i in ids if i < m.n_vocab]

    seq = list(ids)
    gen = []
    for _ in range(args.tokens):
        logits = R.forward(m, seq)
        g = int(np.argmax(logits))
        gen.append(g)
        seq.append(g)
    print(' '.join(str(g) for g in gen))


if __name__ == '__main__':
    main()
