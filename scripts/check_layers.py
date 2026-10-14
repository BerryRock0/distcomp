#!/usr/bin/env python3
"""Проверка раскладки: каждый слой модели помещается в память, которую
объявил хоть один из узлов, и сумма слоёв не превышает их суммарный объём.

Аргументы: файл модели, затем через запятую лимиты узлов в байтах.
"""
import struct
import sys

data = open(sys.argv[1], 'rb').read()
limits = [int(x) for x in sys.argv[2].split(',')]
off = 0
sizes = []
while off + 20 <= len(data):
    _idx, dim_in, dim_out, _act = struct.unpack_from('<IIII', data, off + 4)
    size = 20 + (dim_out * dim_in + dim_out) * 4
    sizes.append(size)
    off += size

if off != len(data):
    print('остаток в файле не кратен слою', file=sys.stderr)
    sys.exit(1)
if not sizes:
    print('в модели нет слоёв', file=sys.stderr)
    sys.exit(1)

# раскладка возможна, если слои укладываются в суммарную память и каждый
# слой влезает хотя бы в один узел
if max(sizes) > max(limits):
    print('слой %d байт не влезает ни в один узел (максимум %d)' % (max(sizes), max(limits)),
          file=sys.stderr)
    sys.exit(1)
if sum(sizes) > sum(limits):
    print('модель %d байт не влезает в %d байт памяти узлов' % (sum(sizes), sum(limits)),
          file=sys.stderr)
    sys.exit(1)

print('слоёв %d, по %d байт, всего %d из %d байт памяти' %
      (len(sizes), sizes[0], sum(sizes), sum(limits)))