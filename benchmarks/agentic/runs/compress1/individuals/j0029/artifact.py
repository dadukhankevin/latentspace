"""Adaptive context-mixed arithmetic coding with long-context phrase rows."""

_TOP = 0xFFFFFFFF
_MID = 0x80000000
_LO = 0x40000000
_HI = 0xC0000000


class _BitStream:
    def __init__(self, raw=b""):
        self.data = bytearray(raw)
        self.acc = 0
        self.count = 0
        self.at = 0

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

    def get(self):
        if self.at >= len(self.data) * 8:
            return 0
        bit = (self.data[self.at >> 3] >> (7 - (self.at & 7))) & 1
        self.at += 1
        return bit


def _text_prior():
    p = [1] * 256
    for x in range(32, 127):
        p[x] = 2
    for x in b" etaoinshrdlucmfwypvbgkjqxz":
        p[x] += 7
    for x in b"ETAOINSHRDLU":
        p[x] += 3
    p[10] += 20
    p[9] += 3
    return p


class _Model:
    """The arithmetic alphabet is always all bytes; context rows add mass."""

    def __init__(self):
        self.global_counts = _text_prior()
        self.global_total = sum(self.global_counts)
        self.one = [[1] * 256 for _ in range(256)]
        self.one_total = [256] * 256
        # Each sparse row is [mass, {byte: count}].  The first three rows are
        # the PPM orders in the base method; the last two are phrase rows.
        self.rows = [{}, {}, {}, {}, {}, {}]
        self.seen = 0
        self.h2 = 0
        self.h3 = 0
        self.h4 = 0
        self.h6 = 0
        self.h8 = 0
        self.h10 = 0

    @staticmethod
    def _scaled(value, total):
        q = (value * 256 + (total >> 1)) // total
        return q if q else 1

    @staticmethod
    def _add_row(table, key, symbol):
        row = table.get(key)
        if row is None:
            table[key] = [1, {symbol: 1}]
            return
        row[0] += 1
        counts = row[1]
        counts[symbol] = counts.get(symbol, 0) + 1
        if row[0] > 768:
            total = 0
            for x in list(counts):
                v = (counts[x] + 1) >> 1
                if v:
                    counts[x] = v
                    total += v
                else:
                    del counts[x]
            row[0] = total

    @staticmethod
    def _mix_row(freq, row, weight):
        if row is None:
            return
        mass, counts = row
        # A singleton high-order row is useful evidence for a repeated
        # phrase.  Capping the multiplier keeps one bad match recoverable.
        w = weight * min(32, mass)
        for symbol, count in counts.items():
            freq[symbol] += w * _Model._scaled(count, mass)

    def cumulative(self):
        freq = [1] * 256

        # The global model keeps arbitrary bytes cheap, including binary and
        # short inputs where no context has yet been learned.
        for symbol in range(256):
            freq[symbol] += 2 * self._scaled(
                self.global_counts[symbol], self.global_total)

        if self.seen:
            row = self.one[self.h2 & 255]
            total = self.one_total[self.h2 & 255]
            for symbol in range(256):
                freq[symbol] += 7 * self._scaled(row[symbol], total)

        if self.seen >= 2:
            self._mix_row(freq, self.rows[0].get(self.h2), 3)
        if self.seen >= 3:
            self._mix_row(freq, self.rows[1].get(self.h3), 5)
        if self.seen >= 4:
            self._mix_row(freq, self.rows[2].get(self.h4), 8)
        if self.seen >= 6:
            self._mix_row(freq, self.rows[3].get(self.h6), 12)
        if self.seen >= 8:
            self._mix_row(freq, self.rows[4].get(self.h8), 18)
        if self.seen >= 10:
            self._mix_row(freq, self.rows[5].get(self.h10), 24)

        cumulative = [0] * 257
        total = 0
        for symbol, value in enumerate(freq):
            total += value
            cumulative[symbol + 1] = total
        return cumulative

    def update(self, symbol):
        self.global_counts[symbol] += 4
        self.global_total += 4
        if self.global_total > 131072:
            total = 0
            for x in range(256):
                self.global_counts[x] = (self.global_counts[x] + 1) >> 1
                total += self.global_counts[x]
            self.global_total = total

        if self.seen:
            row = self.one[self.h2 & 255]
            row[symbol] += 8
            total = self.one_total[self.h2 & 255] + 8
            if total > 65536:
                total = 0
                for x in range(256):
                    row[x] = (row[x] + 1) >> 1
                    total += row[x]
            self.one_total[self.h2 & 255] = total

        if self.seen >= 2:
            self._add_row(self.rows[0], self.h2, symbol)
        if self.seen >= 3:
            self._add_row(self.rows[1], self.h3, symbol)
        if self.seen >= 4:
            self._add_row(self.rows[2], self.h4, symbol)
        if self.seen >= 6:
            self._add_row(self.rows[3], self.h6, symbol)
        if self.seen >= 8:
            self._add_row(self.rows[4], self.h8, symbol)
        if self.seen >= 10:
            self._add_row(self.rows[5], self.h10, symbol)

        self.h10 = ((self.h10 << 8) | symbol) & 0xFFFFFFFFFFFFFFFFFFFF
        self.h8 = ((self.h8 << 8) | symbol) & 0xFFFFFFFFFFFFFFFF
        self.h6 = ((self.h6 << 8) | symbol) & 0xFFFFFFFFFFFF
        self.h4 = ((self.h4 << 8) | symbol) & 0xFFFFFFFF
        self.h3 = ((self.h3 << 8) | symbol) & 0xFFFFFF
        self.h2 = ((self.h2 << 8) | symbol) & 0xFFFF
        self.seen += 1


def _renorm_write(bits, low, high, pending):
    while True:
        if high < _MID:
            bits.put(0)
            while pending:
                bits.put(1)
                pending -= 1
        elif low >= _MID:
            bits.put(1)
            while pending:
                bits.put(0)
                pending -= 1
            low -= _MID
            high -= _MID
        elif low >= _LO and high < _HI:
            pending += 1
            low -= _LO
            high -= _LO
        else:
            return low, high, pending
        low <<= 1
        high = (high << 1) | 1


def compress(data: bytes) -> bytes:
    bits = _BitStream()
    model = _Model()
    low, high, pending = 0, _TOP, 0
    for symbol in data:
        cumulative = model.cumulative()
        total = cumulative[256]
        span = high - low + 1
        high = low + (span * cumulative[symbol + 1]) // total - 1
        low = low + (span * cumulative[symbol]) // total
        low, high, pending = _renorm_write(bits, low, high, pending)
        model.update(symbol)
    pending += 1
    if low < _LO:
        bits.put(0)
        while pending:
            bits.put(1)
            pending -= 1
    else:
        bits.put(1)
        while pending:
            bits.put(0)
            pending -= 1
    bits.finish()
    return len(data).to_bytes(4, "big") + bytes(bits.data)


def _find(cumulative, target):
    lo, hi = 0, 256
    while lo < hi:
        mid = (lo + hi) >> 1
        if cumulative[mid + 1] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        raise ValueError("short compressed stream")
    size = int.from_bytes(blob[:4], "big")
    bits = _BitStream(blob[4:])
    model = _Model()
    low, high = 0, _TOP
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()

    out = bytearray()
    for _ in range(size):
        cumulative = model.cumulative()
        total = cumulative[256]
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        symbol = _find(cumulative, target)
        high = low + (span * cumulative[symbol + 1]) // total - 1
        low = low + (span * cumulative[symbol]) // total
        while True:
            if high < _MID:
                pass
            elif low >= _MID:
                low -= _MID
                high -= _MID
                value -= _MID
            elif low >= _LO and high < _HI:
                low -= _LO
                high -= _LO
                value -= _LO
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.get()
        out.append(symbol)
        model.update(symbol)
    return bytes(out)
