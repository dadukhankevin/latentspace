"""Whole-message text compressor using BWT, MTF, binary zero runs, and AC."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
Q1 = 0x40000000
Q3 = 0xC0000000
ALPH = 259
END = 258
INCREMENT = 28
LIMIT = 42120


class _Bits:
    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.acc = 0
        self.have = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.have += 1
        if self.have == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.have = 0

    def finish(self):
        while self.have:
            self.put(0)

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        v = self.data[self.pos >> 3]
        bit = (v >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


def _bwt(values):
    """Return the last column and the row containing the original rotation."""
    n = len(values)
    suffix_rank = list(values)
    order = list(range(n))
    width = 1
    while True:
        order.sort(key=lambda i: (suffix_rank[i], suffix_rank[(i + width) % n]))
        new_rank = [0] * n
        classes = 0
        first = order[0]
        prev = (suffix_rank[first], suffix_rank[(first + width) % n])
        new_rank[first] = 0
        for i in order[1:]:
            pair = (suffix_rank[i], suffix_rank[(i + width) % n])
            if pair != prev:
                classes += 1
                prev = pair
            new_rank[i] = classes
        suffix_rank = new_rank
        if classes == n - 1:
            break
        width <<= 1

    last = [0] * n
    primary = 0
    for row, start in enumerate(order):
        if start == 0:
            primary = row
        last[row] = values[(start - 1) % n]
    return last, primary


def _unbwt(last, primary):
    n = len(last)
    counts = [0] * 257
    for c in last:
        counts[c] += 1
    starts = [0] * 257
    total = 0
    for c in range(257):
        starts[c] = total
        total += counts[c]
    seen = [0] * 257
    links = [0] * n
    for row, c in enumerate(last):
        links[row] = starts[c] + seen[c]
        seen[c] += 1
    row = primary
    out = [0] * n
    for pos in range(n - 1, -1, -1):
        out[pos] = last[row]
        row = links[row]
    return list(out)


def _tokens_from_ranks(ranks):
    tokens = []
    run = 0
    for rank in ranks:
        if rank == 0:
            run += 1
            continue
        if run:
            x = run - 1
            while True:
                tokens.append(x & 1)
                x >>= 1
                if not x:
                    break
            run = 0
        tokens.append(rank + 1)
    if run:
        x = run - 1
        while True:
            tokens.append(x & 1)
            x >>= 1
            if not x:
                break
    tokens.append(END)
    return tokens


def _ranks_from_tokens(tokens, count):
    ranks = []
    run_value = 0
    run_power = 1
    for token in tokens:
        if token < 2:
            run_value += token * run_power
            run_power <<= 1
            continue
        if run_power != 1:
            ranks.extend([0] * (run_value + 1))
            run_value = 0
            run_power = 1
        if token == END:
            break
        ranks.append(token - 1)
    if run_power != 1:
        ranks.extend([0] * (run_value + 1))
    if len(ranks) != count:
        raise ValueError("invalid rank stream")
    return ranks


def _rescale(freq):
    if sum(freq) > LIMIT:
        for i, value in enumerate(freq):
            freq[i] = (value + 1) >> 1


def _encode_tokens(tokens):
    bits = _Bits()
    freq = [1] * ALPH
    low = 0
    high = FULL
    pending = 0

    def emit(bit):
        nonlocal pending
        bits.put(bit)
        while pending:
            bits.put(1 - bit)
            pending -= 1

    for symbol in tokens:
        cumulative = sum(freq[:symbol])
        amount = freq[symbol]
        total = sum(freq)
        span = high - low + 1
        high = low + (span * (cumulative + amount)) // total - 1
        low = low + (span * cumulative) // total
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
        freq[symbol] += INCREMENT
        _rescale(freq)
    pending += 1
    emit(0 if low < Q1 else 1)
    bits.finish()
    return bytes(bits.data)


def _decode_tokens(blob, needed):
    bits = _Bits(blob)
    freq = [1] * ALPH
    low = 0
    high = FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    tokens = []
    while True:
        span = high - low + 1
        total = sum(freq)
        target = ((value - low + 1) * total - 1) // span
        cumulative = 0
        symbol = 0
        for symbol, amount in enumerate(freq):
            if target < cumulative + amount:
                break
            cumulative += amount
        amount = freq[symbol]
        high = low + (span * (cumulative + amount)) // total - 1
        low = low + (span * cumulative) // total
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
        freq[symbol] += INCREMENT
        _rescale(freq)
        tokens.append(symbol)
        if symbol == END:
            break
        if len(tokens) > needed * 4 + 1024:
            raise ValueError("token stream is too long")
    return tokens


def compress(data: bytes) -> bytes:
    source = list(data) + [256]
    last, primary = _bwt(source)
    mtf = list(range(257))
    ranks = []
    for symbol in last:
        rank = mtf.index(symbol)
        ranks.append(rank)
        del mtf[rank]
        mtf.insert(0, symbol)
    tokens = _tokens_from_ranks(ranks)
    coded = _encode_tokens(tokens)
    return len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + coded


def decompress(blob: bytes) -> bytes:
    if len(blob) < 8:
        raise ValueError("truncated header")
    n = int.from_bytes(blob[:4], "big")
    primary = int.from_bytes(blob[4:8], "big")
    length = n + 1
    tokens = _decode_tokens(blob[8:], length)
    ranks = _ranks_from_tokens(tokens, length)
    mtf = list(range(257))
    last = []
    for rank in ranks:
        if rank < 0 or rank >= 257:
            raise ValueError("invalid MTF rank")
        symbol = mtf[rank]
        last.append(symbol)
        del mtf[rank]
        mtf.insert(0, symbol)
    values = _unbwt(last, primary)
    if not values or values[-1] != 256:
        raise ValueError("missing sentinel")
    values.pop()
    if len(values) != n:
        raise ValueError("length mismatch")
    return bytes(values)
