"""Export sklearn's load_digits (UCI optical digits, 1,797 images at 8x8) to digits.json.

Each image is a 64-char string; each char indexes '0123456789abcdefg' (pixel value 0-16).
Usage: pip install scikit-learn && python tools/export_digits.py
"""
import json
from sklearn.datasets import load_digits

A = '0123456789abcdefg'
d = load_digits()
out = {
    'source': 'UCI optical recognition of handwritten digits (sklearn load_digits)',
    'n': len(d.target),
    'encoding': 'each image = 64 chars, pixel value 0-16 as index into 0123456789abcdefg',
    'X': [''.join(A[int(v)] for v in row) for row in d.data],
    'y': ''.join(str(int(t)) for t in d.target),
}
with open('digits.json', 'w') as f:
    json.dump(out, f, separators=(',', ':'))
print(out['n'], 'images written to digits.json')
