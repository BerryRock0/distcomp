#!/usr/bin/env python3
"""Спрашивает у compute-узла, что он умеет считать."""
import json
import socket
import struct
import sys
import zlib

MAGIC, VER = 0x4443, 2
REQ, RESP, PING, PONG = 1, 2, 3, 4


def pack(t, body):
    b = body.encode()
    crc = zlib.crc32(b) & 0xFFFFFFFF
    # magic(2) ver(1) type(1) flags(2) len(4) crc(4)
    head = struct.pack('>HBBHI', MAGIC, VER, t, 1 if b else 0, len(b))
    return head + struct.pack('>I', crc) + b


def unpack(d):
    magic, ver, t, flags, ln = struct.unpack('>HBBHI', d[:10])
    if magic != MAGIC:
        raise ValueError('плохая магия')
    ln, crc = struct.unpack('>II', d[6:14])
    body = d[14:14 + ln]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError('CRC не сошёлся')
    return t, body.decode('utf-8', 'replace')


def ask(port, payload, role='compute', timeout=2.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(pack(REQ, json.dumps({'role': role, 'title': 't',
                                       'payload': payload, 'attempt': 1,
                                       'context': []})),
                 ('127.0.0.1', port))
        _, body = unpack(s.recv(65535))
        return json.loads(body)
    except Exception as e:
        return {'ok': False, 'output': '', 'detail': f'ТАЙМАУТ/ОШИБКА: {e}'}
    finally:
        s.close()


if __name__ == '__main__':
    port = int(sys.argv[1])
    cases = [
        ('+ - * /',              '2+3*4'),
        ('скобки и приоритет',    '(2+3)*4'),
        ('% остаток',            '17 % 5'),
        ('^ степень',            '2^10'),
        ('правоассоциативность', '2^3^2'),
        ('унарный минус',        '-5 + 3'),
        ('sqrt',                 'sqrt(144)'),
        ('abs',                  'abs(-7.5)'),
        ('round/floor/ceil',     'round(2.5)+floor(2.9)+ceil(2.1)'),
        ('min/max',              'max(3,9) + min(4,2)'),
        ('pow',                  'pow(2,10)'),
        ('log/ln-like',          'log10(1000)'),
        ('exp',                  'exp(1)'),
        ('тригонометрия',        'sin(0)+cos(0)'),
        ('atan2',                'atan2(1,1)'),
        ('константа pi',         'pi'),
        ('константа e',          'e'),
        ('tau',                  'tau'),
        ('знак',                 'sign(-9)'),
        ('подчёркивание',        '1_000_000 / 4'),
        ('запятые: аргументы',  'max(3, 9) + min(4, 2)'),
        ('дробное',              '1/3'),
        ('научная запись',       '1.5e3 + 2e-2'),
        ('вложенные скобки',     '((2+3)*(4-1))/5'),
    ]
    print(f'{"возможность":<22} {"payload":<32} результат')
    print('-' * 86)
    for name, expr in cases:
        r = ask(port, expr)
        out = r.get('output') or r.get('detail', '')
        print(f'{name:<22} {expr:<32} {out}')

    print()
    print('--- что происходит с НЕ-арифметикой ---')
    for payload in ['посчитать стоимость закупки', '1250 рублей за штуку',
                    'сравни 10 и 20', 'вычисли НОД(48, 18)', 'max(1,2,3)']:
        r = ask(port, payload)
        print(f'{payload:<32} ok={str(r.get("ok")):<5} -> {r.get("output")!r}')
        print(f'{"":<32} {r.get("detail","")[:76]}')
