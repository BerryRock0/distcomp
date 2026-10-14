#!/usr/bin/env python3
"""Печатает состояние по стадиям для сверки с движком на C (engine_stages).
Метки строк совпадают один в один: embed, in<N>, final, logits, argmax."""
import sys
import numpy as np

sys.path.insert(0, 'scripts')
import llama_ref as R


def main():
    m = R.Model().build()
    tokens = [int(x) for x in sys.argv[1:]] or [1]

    def W(name):
        shape, data = m.tensors[name]
        return np.array(data, dtype=np.float32).reshape(shape)

    def rms(x, w, eps):
        ss = (x * x).sum(axis=0) / x.shape[0]
        return x / np.sqrt(ss + eps) * w[:, None]

    def rope(vec, pos):
        out = vec.copy()
        half = m.head_dim // 2
        freqs = 1.0 / (m.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / m.head_dim))
        ang = pos * freqs
        c, s = np.cos(ang), np.sin(ang)
        out[:half] = vec[:half] * c - vec[half:half * 2] * s
        out[half:half * 2] = vec[half:half * 2] * c + vec[:half] * s
        return out

    T = len(tokens)
    x = W('token_embd.weight')[:, tokens]
    print('embed  %s' % ' '.join('%.6f' % v for v in x[:, 0][:6]))

    for i in range(m.n_layer):
        print('in%-4d %s' % (i, ' '.join('%.6f' % v for v in x[:, 0][:6])))
        h = rms(x, W('blk.%d.attn_norm.weight' % i), m.rms_eps)
        q = (W('blk.%d.attn_q.weight' % i).T @ h).reshape(m.n_head, m.head_dim, T)
        k = (W('blk.%d.attn_k.weight' % i).T @ h).reshape(m.n_head_kv, m.head_dim, T)
        v = (W('blk.%d.attn_v.weight' % i).T @ h).reshape(m.n_head_kv, m.head_dim, T)
        out = np.zeros((m.n_head, m.head_dim, T), dtype=np.float32)
        for g in range(m.n_head):
            kvg = g // (m.n_head // m.n_head_kv)
            for t in range(T):
                qt = rope(q[g, :, t], t)
                sc = []
                for t2 in range(t + 1):
                    kt = rope(k[kvg, :, t2], t2)
                    sc.append(float((qt * kt).sum()) / np.sqrt(m.head_dim))
                mx = max(sc)
                ex = [np.exp(s - mx) for s in sc]
                tot = sum(ex)
                acc = np.zeros(m.head_dim, dtype=np.float32)
                for t2 in range(t + 1):
                    acc += (ex[t2] / tot) * v[kvg, :, t2]
                out[g, :, t] = acc
        x = x + W('blk.%d.attn_output.weight' % i).T @ out.reshape(m.n_head * m.head_dim, T)
        print('mid    %s' % ' '.join('%.6f' % v for v in x[:, 0][:6]))
        print('aout   %s' % ' '.join('%.6f' % v for v in (W('blk.%d.attn_output.weight' % i).T @ out.reshape(m.n_head * m.head_dim, T))[:, 0][:6]))
        print('att0   %s' % ' '.join('%.6f' % v for v in out[0, :, 0][:6]))
        h = rms(x, W('blk.%d.ffn_norm.weight' % i), m.rms_eps)
        gate = W('blk.%d.ffn_gate.weight' % i).T @ h
        up = W('blk.%d.ffn_up.weight' % i).T @ h
        print('gate   %s' % ' '.join('%.6f' % v for v in gate[:, 0]))
        print('up     %s' % ' '.join('%.6f' % v for v in up[:, 0]))
        act = (gate / (1.0 + np.exp(-gate))) * up
        print('act    %s' % ' '.join('%.6f' % v for v in act[:, 0]))
        print('down   %s' % ' '.join('%.6f' % v for v in (W('blk.%d.ffn_down.weight' % i).T @ act)[:, 0][:6]))
        x = x + W('blk.%d.ffn_down.weight' % i).T @ act

    print('final  %s' % ' '.join('%.6f' % v for v in x[:, T - 1][:6]))
    xf = rms(x[:, T - 1:], W('output_norm.weight'), m.rms_eps)[:, 0]
    lg = W('output.weight').T @ xf
    print('logits %s' % ' '.join('%.6f' % v for v in lg[:8]))
    print('argmax %d' % int(np.argmax(lg)))


if __name__ == '__main__':
    main()