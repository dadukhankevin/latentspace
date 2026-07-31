import numpy as np


FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1
ALPHABET = 258
INCREMENT = 32
LIMIT = 1 << 16


class _BitOut:
    def __init__(self):
        self.data = bytearray()
        self.acc = 0
        self.count = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.count += 1
        if self.count == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.count = 0

    def finish(self):
        while self.count:
            self.put(0)


class _BitIn:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self):
        if self.pos < len(self.data) * 8:
            ans = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        else:
            ans = 0
        self.pos += 1
        return ans


class _Model:
    def __init__(self):
        self.freq = np.ones(ALPHABET, dtype=np.int64)

    def update(self, symbol):
        self.freq[symbol] += INCREMENT
        if int(self.freq.sum()) > LIMIT:
            self.freq = (self.freq + 1) // 2

    def span(self, symbol):
        before = int(self.freq[:symbol].sum())
        return before, int(self.freq[symbol]), int(self.freq.sum())

    def locate(self, target):
        cumulative = np.cumsum(self.freq)
        symbol = int(np.searchsorted(cumulative, target, side="right"))
        before = int(cumulative[symbol - 1]) if symbol else 0
        return symbol, before, int(self.freq[symbol]), int(cumulative[-1])


def _bwt(data):
    n = len(data)
    if n == 0:
        return b"", 0
    suffixes = list(range(n))
    ranks = list(data)
    step = 1
    while step < n:
        suffixes.sort(key=lambda i: (ranks[i], ranks[(i + step) % n]))
        next_ranks = [0] * n
        classes = 0
        previous = None
        for row, start in enumerate(suffixes):
            key = (ranks[start], ranks[(start + step) % n])
            if row and key != previous:
                classes += 1
            next_ranks[start] = classes
            previous = key
        ranks = next_ranks
        if classes == n - 1:
            break
        step <<= 1
    last = bytes(data[(start - 1) % n] for start in suffixes)
    return last, suffixes.index(0)


def _ibwt(last, primary):
    n = len(last)
    if n == 0:
        return b""
    counts = [0] * 256
    occurrence = [0] * n
    for i, symbol in enumerate(last):
        occurrence[i] = counts[symbol]
        counts[symbol] += 1
    starts = [0] * 256
    total = 0
    for symbol in range(256):
        starts[symbol] = total
        total += counts[symbol]
    answer = bytearray(n)
    row = primary
    for i in range(n - 1, -1, -1):
        symbol = last[row]
        answer[i] = symbol
        row = starts[symbol] + occurrence[row]
    return bytes(answer)


def _mtf_encode(last):
    table = list(range(256))
    ranks = []
    for symbol in last:
        rank = table.index(symbol)
        ranks.append(rank)
        table.pop(rank)
        table.insert(0, symbol)
    return ranks


def _mtf_decode(ranks):
    table = list(range(256))
    answer = bytearray()
    for rank in ranks:
        symbol = table[rank]
        answer.append(symbol)
        table.pop(rank)
        table.insert(0, symbol)
    return bytes(answer)


def _rle_tokens(ranks):
    # Symbols 0 and 1 are the two digits of a bijective binary zero run.
    # Symbols 2..256 are MTF ranks 1..255; 257 terminates the stream.
    tokens = []
    i = 0
    while i < len(ranks):
        if ranks[i] == 0:
            end = i + 1
            while end < len(ranks) and ranks[end] == 0:
                end += 1
            value = end - i - 1
            while True:
                tokens.append(1 if value & 1 else 0)
                if value < 2:
                    break
                value = (value - 2) >> 1
            i = end
        else:
            tokens.append(ranks[i] + 1)
            i += 1
    tokens.append(257)
    return tokens


def _encode_tokens(tokens):
    bits = _BitOut()
    model = _Model()
    low, high, pending = 0, FULL, 0

    def emit(bit):
        nonlocal pending
        bits.put(bit)
        while pending:
            bits.put(1 - bit)
            pending -= 1

    for symbol in tokens:
        cumulative, frequency, total = model.span(symbol)
        width = high - low + 1
        high = low + width * (cumulative + frequency) // total - 1
        low = low + width * cumulative // total
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
        model.update(symbol)
    pending += 1
    emit(0 if low < Q1 else 1)
    bits.finish()
    return bytes(bits.data)


def _decode_tokens(blob):
    bits = _BitIn(blob)
    model = _Model()
    low, high, value = 0, FULL, 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    tokens = []
    while True:
        width = high - low + 1
        total = int(model.freq.sum())
        target = ((value - low + 1) * total - 1) // width
        symbol, cumulative, frequency, total = model.locate(target)
        high = low + width * (cumulative + frequency) // total - 1
        low = low + width * cumulative // total
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
            value = (value << 1) | bits.get()
        model.update(symbol)
        tokens.append(symbol)
        if symbol == 257:
            return tokens


def compress(data: bytes) -> bytes:
    if not data:
        return (0).to_bytes(4, "big") + (0).to_bytes(4, "big")
    last, primary = _bwt(data)
    ranks = _mtf_encode(last)
    stream = _encode_tokens(_rle_tokens(ranks))
    return (len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + stream)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    primary = int.from_bytes(blob[4:8], "big")
    if n == 0:
        return b""
    tokens = _decode_tokens(blob[8:])
    ranks = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token < 2:
            run = 0
            place = 1
            while token < 2:
                run += (token + 1) * place
                place <<= 1
                i += 1
                token = tokens[i]
            ranks.extend([0] * run)
        if token == 257:
            break
        ranks.append(token - 1)
        i += 1
    last = _mtf_decode(ranks)
    return _ibwt(last, primary)[:n]
