#!/usr/bin/env python3
"""Модель с меняющейся шириной: 4 -> 6 -> 8 -> 5.

Служит испытанием для раскладки: у настоящих моделей ширина растёт от слоя
к слою, и любая буферная ошибка (обрезанный массив активаций, неверная
ширина на выходе) тут же всплывает — сверка с целиком посчитанной моделью
даёт ненулевое расхождение.

Файл: DCSH | u32 index | u32 dim_in | u32 dim_out | u32 act | W | b
"""
import random
import struct
import sys

LAYERS = [(4, 6), (6, 8), (8, 5)]

def build(path, seed=5):
    rnd = random.Random(seed)
    out = b''
    for idx, (dim_in, dim_out) in enumerate(LAYERS):
        w = [rnd.uniform(-0.5, 0.5) for _ in range(dim_in * dim_out)]
        b = [rnd.uniform(-0.2, 0.2) for _ in range(dim_out)]
        out += b'DCSH' + struct.pack('<IIII', idx, dim_in, dim_out, 0)
        out += struct.pack('<%df' % len(w), *w)
        out += struct.pack('<%df' % len(b), *b)
    open(path, 'wb').write(out)
    return len(out)


if __name__ == '__main__':
    n = build(sys.argv[1])
    print('модель с растущей шириной: %s, %d байт' %
          (' -> '.join('%d' % d for _i, d in LAYERS), n))