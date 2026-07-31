"""Adaptive byte arithmetic coder with interpolated suffix contexts."""

_FULL = 0xFFFFFFFF
_HALF = 0x80000000
_Q1 = 0x40000000
_Q3 = 0xC0000000
_SCALE = 256


def _make_prior():
    # A fixed, deliberately generic text prior.  It is never changed by the
    # stream, and the adaptive rows take over as soon as evidence accumulates.
    p = [1] * 256
    for x in range(32, 127):
        p[x] = 2
    for x in b" etaoinshrdlucmfwypvbgkjqxz":
        p[x] += 8
    for x in b"ETAOINSHRDLU":
        p[x] += 4
    p[10] += 24
    p[9] += 4
    return p


class _Bits:
    def __init__(self, data=b""):
        self.buf = bytearray(data)
        self.acc = 0
        self.nbits = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def finish(self):
        while self.nbits:
            self.put(0)

    def get(self):
        if self.pos >= len(self.buf) * 8:
            return 0
        v = (self.buf[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return v


class _Model:
    def __init__(self):
        self.prior = _make_prior()
        self.prior_total = sum(self.prior)
        self.rows1 = [[1] * 256 for _ in range(256)]
        self.totals1 = [256] * 256
        # A row is [total, {next_byte: count}].  Keeping these rows sparse is
        # important: most long contexts occur only once in natural text.
        self.tables = [{}, {}, {}]
        self.nseen = 0
        self.h1 = 0
        self.h2 = 0
        self.h3 = 0
        self.h4 = 0

    @staticmethod
    def _row_mass(row):
        return row[0] if row is not None else 0

    def _put_in_row(self, table, key, symbol):
        row = table.get(key)
        if row is None:
            table[key] = [1, {symbol: 1}]
            return
        row[0] += 1
        d = row[1]
        d[symbol] = d.get(symbol, 0) + 1
        # Contexts are local models.  Aging prevents a long-lived repeated
        # phrase from becoming an inflexible permanent dictionary.
        if row[0] > 512:
            total = 0
            for k in list(d):
                v = (d[k] + 1) >> 1
                if v:
                    d[k] = v
                    total += v
                else:
                    del d[k]
            row[0] = total

    def _q(self, value, total):
        q = (value * _SCALE + (total >> 1)) // total
        return q if q > 0 else 1

    def cumulative(self):
        # Interpolate normalized component distributions into integer counts.
        # The weights are evidence-sensitive rather than a fixed order bias.
        r1 = self.rows1[self.h1] if self.nseen else None
        t1 = self.totals1[self.h1] if self.nseen else 0
        w0 = 2
        w1 = 8
        freq = [w0 * self._q(self.prior[x], self.prior_total) + 1
                for x in range(256)]
        if r1 is not None:
            for x in range(256):
                freq[x] += w1 * self._q(r1[x], t1)

        if self.nseen >= 2:
            row = self.tables[0].get(self.h2)
            if row is not None:
                mass, d = row
                w = min(32, mass)
                for x, v in d.items():
                    freq[x] += w * self._q(v, mass)
        if self.nseen >= 3:
            row = self.tables[1].get(self.h3)
            if row is not None:
                mass, d = row
                w = 2 * min(32, mass)
                for x, v in d.items():
                    freq[x] += w * self._q(v, mass)
        if self.nseen >= 4:
            row = self.tables[2].get(self.h4)
            if row is not None:
                mass, d = row
                w = 4 * min(32, mass)
                for x, v in d.items():
                    freq[x] += w * self._q(v, mass)

        cumulative = [0] * 257
        total = 0
        for x in range(256):
            total += freq[x]
            cumulative[x + 1] = total
        return cumulative

    def update(self, symbol):
        if self.nseen:
            row = self.rows1[self.h1]
            row[symbol] += 17
            total = self.totals1[self.h1] + 17
            if total > 60000:
                total = 0
                for x in range(256):
                    row[x] = (row[x] + 1) >> 1
                    total += row[x]
            self.totals1[self.h1] = total
        if self.nseen >= 2:
            self._put_in_row(self.tables[0], self.h2, symbol)
        if self.nseen >= 3:
            self._put_in_row(self.tables[1], self.h3, symbol)
        if self.nseen >= 4:
            self._put_in_row(self.tables[2], self.h4, symbol)

        self.h4 = ((self.h4 << 8) | symbol) & 0xFFFFFFFF
        self.h3 = ((self.h3 << 8) | symbol) & 0xFFFFFF
        self.h2 = ((self.h2 << 8) | symbol) & 0xFFFF
        self.h1 = symbol
        self.nseen += 1


def _emit_interval(out, low, high, pending):
    while True:
        if high < _HALF:
            out.put(0)
            while pending:
                out.put(1)
                pending -= 1
        elif low >= _HALF:
            out.put(1)
            while pending:
                out.put(0)
                pending -= 1
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
    return low, high, pending


def compress(data: bytes) -> bytes:
    out = _Bits()
    model = _Model()
    low, high, pending = 0, _FULL, 0
    for symbol in data:
        c = model.cumulative()
        total = c[256]
        span = high - low + 1
        left = c[symbol]
        right = c[symbol + 1]
        high = low + (span * right) // total - 1
        low = low + (span * left) // total
        low, high, pending = _emit_interval(out, low, high, pending)
        model.update(symbol)
    pending += 1
    if low < _Q1:
        out.put(0)
        while pending:
            out.put(1)
            pending -= 1
    else:
        out.put(1)
        while pending:
            out.put(0)
            pending -= 1
    out.finish()
    return len(data).to_bytes(4, "big") + bytes(out.buf)


def _locate(cumulative, target):
    lo = 0
    hi = 256
    while lo < hi:
        mid = (lo + hi) >> 1
        if cumulative[mid + 1] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        raise ValueError("short blob")
    n = int.from_bytes(blob[:4], "big")
    bits = _Bits(blob[4:])
    model = _Model()
    low, high = 0, _FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    result = bytearray()
    for _ in range(n):
        c = model.cumulative()
        total = c[256]
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        symbol = _locate(c, target)
        left = c[symbol]
        right = c[symbol + 1]
        high = low + (span * right) // total - 1
        low = low + (span * left) // total
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
        model.update(symbol)
    return bytes(result)
