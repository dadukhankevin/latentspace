"""Whole-message BWT/MTF compressor with a small arithmetic coder."""

_FULL = 0xFFFFFFFF
_HALF = 1 << 31
_Q1 = 1 << 30
_Q3 = _HALF + _Q1
_ALPHABET = 512
_INC = 19
_LIMIT = 1 << 16


class _Bits:
    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.acc = 0
        self.nbits = 0
        self.pos = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def flush(self):
        while self.nbits:
            self.write(0)

    def read(self):
        if self.pos >= len(self.data) * 8:
            return 0
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


def _suffix_order(data):
    """Sort all cyclic rotations by prefix doubling."""
    n = len(data)
    order = list(range(n))
    rank = list(data)
    step = 1
    while step < n:
        order.sort(key=lambda i: (rank[i], rank[(i + step) % n]))
        new_rank = [0] * n
        classes = 0
        old = (rank[order[0]], rank[(order[0] + step) % n])
        new_rank[order[0]] = 0
        for i in order[1:]:
            cur = (rank[i], rank[(i + step) % n])
            if cur != old:
                classes += 1
                old = cur
            new_rank[i] = classes
        rank = new_rank
        if classes == n - 1:
            break
        step <<= 1
    return order


def _bwt(data):
    n = len(data)
    order = _suffix_order(data)
    last = bytearray(n)
    primary = 0
    for row, start in enumerate(order):
        if start == 0:
            primary = row
            last[row] = data[n - 1]
        else:
            last[row] = data[start - 1]
    return bytes(last), primary


def _unbwt(last, primary):
    n = len(last)
    seen = [0] * 256
    occurrence = [0] * n
    for i, symbol in enumerate(last):
        occurrence[i] = seen[symbol]
        seen[symbol] += 1
    first = [0] * 256
    total = 0
    for symbol in range(256):
        first[symbol] = total
        total += seen[symbol]
    result = bytearray(n)
    row = primary
    for i in range(n - 1, -1, -1):
        symbol = last[row]
        result[i] = symbol
        row = first[symbol] + occurrence[row]
    return bytes(result)


def _mtf(data):
    book = list(range(256))
    ranks = []
    for symbol in data:
        rank = book.index(symbol)
        ranks.append(rank)
        if rank:
            book.pop(rank)
            book.insert(0, symbol)
    return ranks


def _unmtf(ranks):
    book = list(range(256))
    result = bytearray()
    for rank in ranks:
        symbol = book[rank]
        result.append(symbol)
        if rank:
            book.pop(rank)
            book.insert(0, symbol)
    return bytes(result)


def _tokens(ranks):
    """Ranks 1..255 are literals; 256+length represents a zero run."""
    tokens = []
    i = 0
    n = len(ranks)
    while i < n:
        rank = ranks[i]
        if rank:
            tokens.append(rank)
            i += 1
            continue
        j = i + 1
        while j < n and ranks[j] == 0:
            j += 1
        run = j - i
        while run > 255:
            tokens.append(511)
            run -= 255
        if run:
            tokens.append(256 + run)
        i = j
    return tokens


def _emit_arithmetic(tokens):
    out = _Bits()
    freq = [0] + [1] * 255 + [0] + [1] * 255
    low = 0
    high = _FULL
    pending = 0

    def emit(bit):
        nonlocal pending
        out.write(bit)
        while pending:
            out.write(1 - bit)
            pending -= 1

    for symbol in tokens:
        total = sum(freq)
        cumulative = sum(freq[:symbol])
        count = freq[symbol]
        span = high - low + 1
        high = low + span * (cumulative + count) // total - 1
        low = low + span * cumulative // total
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _Q1 and high < _Q3:
                pending += 1
                low -= _Q1
                high -= _Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        freq[symbol] += _INC
        if sum(freq) > _LIMIT:
            freq = [(x + 1) // 2 for x in freq]

    pending += 1
    emit(0 if low < _Q1 else 1)
    out.flush()
    return bytes(out.data)


def _decode_arithmetic(blob, target_length):
    bits = _Bits(blob)
    freq = [0] + [1] * 255 + [0] + [1] * 255
    low = 0
    high = _FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.read()

    ranks = []
    while len(ranks) < target_length:
        total = sum(freq)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        cumulative = 0
        symbol = 0
        for symbol, count in enumerate(freq):
            if target < cumulative + count:
                break
            cumulative += count
        count = freq[symbol]
        high = low + span * (cumulative + count) // total - 1
        low = low + span * cumulative // total
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _Q1 and high < _Q3:
                low -= _Q1
                high -= _Q1
                value -= _Q1
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.read()
        freq[symbol] += _INC
        if sum(freq) > _LIMIT:
            freq = [(x + 1) // 2 for x in freq]
        if symbol >= 256:
            ranks.extend([0] * (symbol - 256))
        else:
            ranks.append(symbol)
    return ranks[:target_length]


def compress(data: bytes) -> bytes:
    if not data:
        return b"\x00" * 8
    last, primary = _bwt(data)
    ranks = _mtf(last)
    tokens = _tokens(ranks)
    coded = _emit_arithmetic(tokens)
    return len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + coded


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    if n == 0:
        return b""
    primary = int.from_bytes(blob[4:8], "big")
    tokens = _decode_arithmetic(blob[8:], n)
    ranks = []
    for token in tokens:
        if token >= 256:
            ranks.extend([0] * (token - 256))
        else:
            ranks.append(token)
    ranks = ranks[:n]
    last = _unmtf(ranks)
    return _unbwt(last, primary)
