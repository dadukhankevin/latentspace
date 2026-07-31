"""A BWT/MTF compressor with locally reset adaptive arithmetic models."""

_ALPHABET = 256
_BLOCK = 1 << 30
_TOP = 1 << 16
_INC = 12
_FULL = 0xFFFFFFFF
_HALF = 1 << 31
_Q1 = 1 << 30
_Q3 = _Q1 + _HALF


class _BitsOut:
    def __init__(self):
        self.data = bytearray()
        self.acc = 0
        self.used = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.used += 1
        if self.used == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.used = 0

    def finish(self):
        if self.used:
            self.data.append(self.acc << (8 - self.used))
        return bytes(self.data)


class _BitsIn:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        v = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return v


def _fresh():
    return [1] * _ALPHABET


def _bwt(data):
    n = len(data)
    if not n:
        return b"", 0
    order = list(range(n))
    rank = list(data)
    step = 1
    while step < n:
        width = max(rank) + 1
        order.sort(key=lambda i: rank[i] * width + rank[i + step if i + step < n else i + step - n])
        new_rank = [0] * n
        classes = 0
        previous = None
        for i in order:
            second_at = i + step
            if second_at >= n:
                second_at -= n
            pair = (rank[i], rank[second_at])
            if previous is not None and pair != previous:
                classes += 1
            new_rank[i] = classes
            previous = pair
        rank = new_rank
        if classes == n - 1:
            break
        step <<= 1
    order.sort(key=rank.__getitem__)
    primary = order.index(0)
    last = bytearray(n)
    for j, start in enumerate(order):
        last[j] = data[start - 1 if start else n - 1]
    return bytes(last), primary


def _unbwt(last, primary):
    n = len(last)
    if not n:
        return b""
    counts = [0] * 256
    for c in last:
        counts[c] += 1
    starts = [0] * 256
    running = 0
    for c in range(256):
        starts[c] = running
        running += counts[c]
    seen = [0] * 256
    lf = [0] * n
    for row, c in enumerate(last):
        lf[row] = starts[c] + seen[c]
        seen[c] += 1
    out = bytearray(n)
    row = primary
    for pos in range(n - 1, -1, -1):
        out[pos] = last[row]
        row = lf[row]
    return bytes(out)


def _mtf_tokens(last):
    table = list(range(256))
    tokens = []
    i = 0
    while i < len(last):
        c = last[i]
        rank = table.index(c)
        if rank == 0:
            run = 1
            while i + run < len(last) and table[0] == last[i + run] and run < 256:
                run += 1
            tokens.append(0)
            tokens.append(run - 1)
            i += run
        else:
            tokens.append(rank)
            v = table.pop(rank)
            table.insert(0, v)
            i += 1
    return tokens


def _arith_encode(symbols):
    bits = _BitsOut()
    freq = _fresh()
    low = 0
    high = _FULL
    pending = 0

    def emit(bit):
        nonlocal pending
        bits.put(bit)
        while pending:
            bits.put(1 - bit)
            pending -= 1

    for index, symbol in enumerate(symbols):
        if index and index % _BLOCK == 0:
            freq = _fresh()
        total = sum(freq)
        cum = 0
        for j in range(symbol):
            cum += freq[j]
        span = high - low + 1
        high = low + (span * (cum + freq[symbol]) // total) - 1
        low = low + (span * cum // total)
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
        if total + _INC > _TOP:
            freq = [(v + 1) // 2 for v in freq]

    pending += 1
    emit(0 if low < _Q1 else 1)
    return bits.finish()


def _arith_decode(data, count):
    bits = _BitsIn(data)
    freq = _fresh()
    low = 0
    high = _FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    result = []
    for index in range(count):
        if index and index % _BLOCK == 0:
            freq = _fresh()
        total = sum(freq)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        cum = 0
        symbol = 0
        while cum + freq[symbol] <= target:
            cum += freq[symbol]
            symbol += 1
        sym_freq = freq[symbol]
        high = low + (span * (cum + sym_freq) // total) - 1
        low = low + (span * cum // total)
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
            value = (value << 1) | bits.get()
        result.append(symbol)
        freq[symbol] += _INC
        if total + _INC > _TOP:
            freq = [(v + 1) // 2 for v in freq]
    return result


def _pack_u32(v):
    return bytes(((v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255, v & 255))


def compress(data: bytes) -> bytes:
    last, primary = _bwt(data)
    tokens = _mtf_tokens(last)
    return _pack_u32(len(data)) + _pack_u32(primary) + _pack_u32(len(tokens)) + _arith_encode(tokens)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[0:4], "big")
    primary = int.from_bytes(blob[4:8], "big")
    token_count = int.from_bytes(blob[8:12], "big")
    tokens = _arith_decode(blob[12:], token_count)
    table = list(range(256))
    last = bytearray()
    i = 0
    while i < len(tokens):
        rank = tokens[i]
        i += 1
        if rank == 0:
            run = tokens[i] + 1
            i += 1
            last.extend(bytes((table[0],)) * run)
        else:
            c = table[rank]
            last.append(c)
            v = table.pop(rank)
            table.insert(0, v)
    if len(last) != n:
        raise ValueError("invalid token stream")
    return _unbwt(bytes(last), primary)
