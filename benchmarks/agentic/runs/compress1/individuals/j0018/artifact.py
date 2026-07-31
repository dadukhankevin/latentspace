"""Deterministic adaptive context arithmetic coder with a match-stream fallback."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
Q1 = 0x40000000
Q3 = 0xC0000000
LIMIT = 10240


class _Bits:
    def __init__(self, raw=b""):
        self.a = bytearray(raw)
        self.x = 0
        self.n = 0
        self.p = 0

    def put(self, bit):
        self.x = (self.x << 1) | bit
        self.n += 1
        if self.n == 8:
            self.a.append(self.x)
            self.x = 0
            self.n = 0

    def finish(self):
        while self.n:
            self.put(0)

    def get(self):
        if self.p >= len(self.a) * 8:
            return 0
        z = (self.a[self.p >> 3] >> (7 - (self.p & 7))) & 1
        self.p += 1
        return z


def _dense_span(row, sym):
    c = 0
    for j in range(sym):
        c += row[j]
    return c, row[sym], sum(row)


def _sparse_span(row, sym):
    c = sym
    for j, v in row.items():
        if j < sym:
            c += v
    return c, row.get(sym, 0) + 1, 256 + sum(row.values())


def _bump(row, sym):
    row[sym] += 40
    if sum(row) > LIMIT:
        for j in range(256):
            row[j] = (row[j] + 1) >> 1


def _global_bump(row, sym):
    row[sym] += 8
    if sum(row) > LIMIT:
        for j in range(256):
            row[j] = (row[j] + 1) >> 1


def _sparse_bump(row, sym):
    row[sym] = row.get(sym, 0) + 84
    if 256 + sum(row.values()) > LIMIT:
        for j in list(row):
            row[j] = (row[j] + 1) >> 1
            if row[j] == 0:
                del row[j]


def _model_state():
    return ([1] * 256, [[1] * 256 for _ in range(256)], [0] * 256,
            {}, {}, {}, {}, {}, {})


def _choice(state, data, i):
    glob, one, one_n, two, two_n, three, three_n, four, four_n = state
    if i >= 4:
        key = ((data[i - 4] << 24) | (data[i - 3] << 16) |
               (data[i - 2] << 8) | data[i - 1])
        n = four_n.get(key, 0)
        row = four.get(key)
        if row is not None and n >= 40:
            return row, True
    if i >= 3:
        key = (data[i - 3] << 16) | (data[i - 2] << 8) | data[i - 1]
        n = three_n.get(key, 0)
        row = three.get(key)
        if row is not None and n >= 6:
            return row, True
    if i >= 2:
        key = (data[i - 2] << 8) | data[i - 1]
        n = two_n.get(key, 0)
        row = two.get(key)
        if row is not None and n >= 24:
            return row, True
    if i:
        prev = data[i - 1]
        if one_n[prev] >= 2:
            return one[prev], False
    return glob, False


def _learn(state, data, i, sym):
    glob, one, one_n, two, two_n, three, three_n, four, four_n = state
    _global_bump(glob, sym)
    if i:
        prev = data[i - 1]
        _bump(one[prev], sym)
        one_n[prev] += 1
    if i >= 2:
        key = (data[i - 2] << 8) | data[i - 1]
        n = two_n.get(key, 0) + 1
        two_n[key] = n
        row = two.get(key)
        if row is None and n == 1 and len(two) < 5000:
            row = {}
            two[key] = row
        if row is not None:
            _sparse_bump(row, sym)
    if i >= 3:
        key = (data[i - 3] << 16) | (data[i - 2] << 8) | data[i - 1]
        n = three_n.get(key, 0) + 1
        three_n[key] = n
        row = three.get(key)
        if row is None and n == 1 and len(three) < 12000:
            row = {}
            three[key] = row
        if row is not None:
            _sparse_bump(row, sym)
    if i >= 4:
        key = ((data[i - 4] << 24) | (data[i - 3] << 16) |
               (data[i - 2] << 8) | data[i - 1])
        n = four_n.get(key, 0) + 1
        four_n[key] = n
        row = four.get(key)
        if row is None and n == 1 and len(four) < 20000:
            row = {}
            four[key] = row
        if row is not None:
            _sparse_bump(row, sym)
def _arith(data):
    bits = _Bits()
    state = _model_state()
    low, high, pending = 0, FULL, 0

    def emit(bit):
        nonlocal pending
        bits.put(bit)
        while pending:
            bits.put(1 - bit)
            pending -= 1

    for i, sym in enumerate(data):
        row, sparse = _choice(state, data, i)
        if sparse:
            cum, freq, total = _sparse_span(row, sym)
        else:
            cum, freq, total = _dense_span(row, sym)
        span = high - low + 1
        high = low + (span * (cum + freq) // total) - 1
        low = low + (span * cum // total)
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
            low = (low << 1) & FULL
            high = ((high << 1) | 1) & FULL
        _learn(state, data, i, sym)
    pending += 1
    emit(0 if low < Q1 else 1)
    bits.finish()
    return bytes(bits.a)


def _unarith(blob, n):
    bits = _Bits(blob)
    state = _model_state()
    low, high, value = 0, FULL, 0
    for _ in range(32):
        value = ((value << 1) | bits.get()) & FULL
    out = bytearray()
    for i in range(n):
        row, sparse = _choice(state, out, i)
        if sparse:
            total = 256 + sum(row.values())
        else:
            total = sum(row)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        if sparse:
            cum = 0
            sym = 0
            for sym in range(256):
                f = row.get(sym, 0) + 1
                if target < cum + f:
                    break
                cum += f
            freq = row.get(sym, 0) + 1
        else:
            cum = 0
            sym = 0
            for sym in range(256):
                f = row[sym]
                if target < cum + f:
                    break
                cum += f
            freq = row[sym]
        high = low + (span * (cum + freq) // total) - 1
        low = low + (span * cum // total)
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
            low = (low << 1) & FULL
            high = ((high << 1) | 1) & FULL
            value = ((value << 1) | bits.get()) & FULL
        out.append(sym)
        _learn(state, out, i, sym)
    return bytes(out)


class _OutBits:
    def __init__(self):
        self.a = bytearray()
        self.x = 0
        self.n = 0

    def put(self, bit):
        self.x = (self.x << 1) | bit
        self.n += 1
        if self.n == 8:
            self.a.append(self.x)
            self.x = 0
            self.n = 0

    def putn(self, x, n):
        for j in range(n - 1, -1, -1):
            self.put((x >> j) & 1)

    def finish(self):
        if self.n:
            self.putn(0, 8 - self.n)


def _lz(data):
    n = len(data)
    prev = {}
    matches = [None] * n
    for i in range(n - 3):
        key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        cand = prev.get(key)
        best = None
        if cand:
            for j in reversed(cand[-24:]):
                lim = min(258, n - i)
                k = 3
                while k < lim and data[j + k] == data[i + k]:
                    k += 1
                if k >= 4 and (best is None or k > best[1]):
                    best = (i - j, k)
        if best is not None:
            matches[i] = best
        if cand is None:
            prev[key] = [i]
        else:
            cand.append(i)
            if len(cand) > 32:
                del cand[0]

    dp = [0] * (n + 1)
    take = [0] * n
    for i in range(n - 1, -1, -1):
        val = 9 + dp[i + 1]
        take[i] = 0
        m = matches[i]
        if m is not None:
            dist, lim = m
            for length in range(4, lim + 1):
                z = 25 + dp[i + length]
                if z < val:
                    val = z
                    take[i] = (dist << 9) | length
        dp[i] = val

    out = _OutBits()
    i = 0
    while i < n:
        t = take[i]
        if t:
            dist, length = t >> 9, t & 511
            out.put(1)
            out.putn(dist - 1, 16)
            out.putn(length - 4, 8)
            i += length
        else:
            out.put(0)
            out.putn(data[i], 8)
            i += 1
    out.finish()
    return bytes(out.a)


def _unlz(blob, n):
    bits = _Bits(blob)
    out = bytearray()
    while len(out) < n:
        if bits.get():
            dist = 0
            for _ in range(16):
                dist = (dist << 1) | bits.get()
            dist += 1
            length = 0
            for _ in range(8):
                length = (length << 1) | bits.get()
            length += 4
            start = len(out) - dist
            for _ in range(length):
                out.append(out[start])
                start += 1
        else:
            x = 0
            for _ in range(8):
                x = (x << 1) | bits.get()
            out.append(x)
    return bytes(out)


def _length_bytes(n):
    out = bytearray()
    while True:
        q = n & 127
        n >>= 7
        out.append(q | (128 if n else 0))
        if not n:
            return bytes(out)


def _read_length(blob):
    n = 0
    shift = 0
    i = 1
    while True:
        q = blob[i]
        i += 1
        n |= (q & 127) << shift
        if q < 128:
            return n, i
        shift += 7


def compress(data: bytes) -> bytes:
    a = _arith(data)
    l = _lz(data)
    # One byte selects the representation; the varint length is not inferred
    # from arithmetic or bit-stream padding.
    header = _length_bytes(len(data))
    if len(a) <= len(l):
        return b"A" + header + a
    return b"L" + header + l


def decompress(blob: bytes) -> bytes:
    n, start = _read_length(blob)
    if blob[:1] == b"A":
        return _unarith(blob[start:], n)
    if blob[:1] == b"L":
        return _unlz(blob[start:], n)
    raise ValueError("unknown representation")
