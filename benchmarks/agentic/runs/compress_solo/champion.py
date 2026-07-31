"""Baseline artifact for the compress task: adaptive order-0 arithmetic
coding. No context modeling at all — it learns only the running byte
frequencies, so it approaches the corpus's unigram entropy (~4.7 bits
per byte). Everything smarter — context mixing, match models, learned
structure — is headroom for evolution. Pure python + numpy, and it
round-trips exactly (the scorer verifies)."""
import numpy as np

INC = 128
PRIOR_INC = 64
UNIGRAM_INC = 128
BACKOFF = 512
UNIGRAM_BACKOFF = 256
LIMIT = 1 << 14
FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1


class Bits:
    def __init__(self, data=b""):
        self.bytes_ = bytearray(data)
        self.acc = 0
        self.n = 0
        self.pos = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.n += 1
        if self.n == 8:
            self.bytes_.append(self.acc)
            self.acc = self.n = 0

    def flush(self):
        while self.n:
            self.write(0)

    def read(self):
        if self.pos < len(self.bytes_) * 8:
            byte = self.bytes_[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            self.pos += 1
            return bit
        return 0


class Model:
    def __init__(self):
        self.freq = np.ones((1 << 16, 256), dtype=np.int64)
        self.freq3 = {}
        self.order1 = np.ones((256, 256), dtype=np.int64)
        self.order0 = np.ones(256, dtype=np.int64)

    def row(self, ctx, ctx3):
        prior = self.order1[ctx & 255]
        total = int(prior.sum())
        prior = (prior * BACKOFF + total // 2) // total
        unigram = self.order0
        unigram_total = int(unigram.sum())
        unigram = (unigram * UNIGRAM_BACKOFF + unigram_total // 2) // unigram_total
        row = self.freq[ctx] + prior + unigram
        extra = self.freq3.get(ctx3)
        if extra is not None:
            row = row + extra
        return row

    def spans(self, ctx, ctx3, s):
        row = self.row(ctx, ctx3)
        cum = int(row[:s].sum())
        return cum, int(row[s]), int(row.sum())

    def find(self, ctx, ctx3, target):
        c = np.cumsum(self.row(ctx, ctx3))
        s = int(np.searchsorted(c, target, side="right"))
        cum = int(c[s - 1]) if s else 0
        row = self.row(ctx, ctx3)
        return s, cum, int(row[s]), int(c[-1])

    def update(self, ctx, ctx3, s):
        self.freq[ctx, s] += INC
        if self.freq[ctx].sum() > LIMIT:
            self.freq[ctx] = (self.freq[ctx] + 1) // 2
        extra = self.freq3.get(ctx3)
        if extra is None:
            extra = np.zeros(256, dtype=np.int64)
            self.freq3[ctx3] = extra
        extra[s] += 7 * INC
        if extra.sum() > LIMIT:
            self.freq3[ctx3] = (extra + 1) // 2
        prev = ctx & 255
        self.order1[prev, s] += PRIOR_INC
        if self.order1[prev].sum() > LIMIT:
            self.order1[prev] = (self.order1[prev] + 1) // 2
        self.order0[s] += UNIGRAM_INC
        if self.order0.sum() > LIMIT:
            self.order0 = (self.order0 + 1) // 2


def compress(data: bytes) -> bytes:
    out = Bits()
    model = Model()
    low, high, pending = 0, FULL, 0
    prev3 = prev2 = prev = 0

    def emit(bit):
        nonlocal pending
        out.write(bit)
        while pending:
            out.write(1 - bit)
            pending -= 1

    for byte in data:
        ctx = (prev2 << 8) | prev
        ctx3 = (prev3 << 16) | ctx
        cum, freq, tot = model.spans(ctx, ctx3, byte)
        span = high - low + 1
        high = low + span * (cum + freq) // tot - 1
        low = low + span * cum // tot
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= Q1 and high < Q3:
                pending += 1
                low -= Q1
                high -= Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        model.update(ctx, ctx3, byte)
        prev3, prev2, prev = prev2, prev, byte
    pending += 1
    emit(0 if low < Q1 else 1)
    out.flush()
    return (len(data) - 1).to_bytes(2, "big") + bytes(out.bytes_)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:2], "big") + 1
    bits = Bits(blob[2:])
    model = Model()
    low, high = 0, FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.read()
    out = bytearray()
    prev3 = prev2 = prev = 0
    for _ in range(n):
        span = high - low + 1
        ctx = (prev2 << 8) | prev
        ctx3 = (prev3 << 16) | ctx
        tot = int(model.row(ctx, ctx3).sum())
        target = ((value - low + 1) * tot - 1) // span
        s, cum, freq, tot = model.find(ctx, ctx3, target)
        high = low + span * (cum + freq) // tot - 1
        low = low + span * cum // tot
        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                value -= HALF
            elif low >= Q1 and high < Q3:
                low -= Q1
                high -= Q1
                value -= Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.read()
        out.append(s)
        model.update(ctx, ctx3, s)
        prev3, prev2, prev = prev2, prev, s
    return bytes(out)
