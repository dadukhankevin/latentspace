"""Adaptive arithmetic coding with a sparse, interpolated context hierarchy."""

_FULL = 0xFFFFFFFF
_HALF = 0x80000000
_Q1 = 0x40000000
_Q3 = 0xC0000000


class _Bits:
    def __init__(self, raw=b""):
        self.data = bytearray(raw)
        self.acc = 0
        self.nbits = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def finish(self):
        while self.nbits:
            self.put(0)


class _Model:
    # Long-context evidence is scaled by the active order-1 mass, so a
    # single reliable continuation remains meaningful against the backoff.
    W0, W1 = 1, 8

    def __init__(self):
        self.zero = [1] * 256
        self.zero_total = 256
        self.one = [[1] * 256 for _ in range(256)]
        self.one_total = [256] * 256
        self.two = {}
        self.two_total = {}
        self.three = {}
        self.three_total = {}
        self.four = {}
        self.four_total = {}

    @staticmethod
    def _prefix(row, symbol):
        if row is None:
            return 0
        return sum(v for k, v in row.items() if k < symbol)

    def _parts(self, history):
        return (self.one[history & 255],
                self.two.get(history & 65535),
                self.three.get(history & 0xFFFFFF),
                self.four.get(history & 0xFFFFFFFF))

    def distribution(self, history, symbol=None, target=None):
        r1, r2, r3, r4 = self._parts(history)
        w2 = self.one_total[history & 255]
        w3 = w2 * 2
        w4 = w2 * 4
        n2 = self.two_total.get(history & 65535, 0)
        n3 = self.three_total.get(history & 0xFFFFFF, 0)
        n4 = self.four_total.get(history & 0xFFFFFFFF, 0)
        total = (self.W0 * self.zero_total +
                 self.W1 * self.one_total[history & 255] +
                 w2 * n2 + w3 * n3 + w4 * n4)

        if symbol is not None:
            freq = (self.W0 * self.zero[symbol] +
                    self.W1 * r1[symbol] +
                    w2 * (r2.get(symbol, 0) if r2 else 0) +
                    w3 * (r3.get(symbol, 0) if r3 else 0) +
                    w4 * (r4.get(symbol, 0) if r4 else 0))
            cum = (self.W0 * sum(self.zero[:symbol]) +
                   self.W1 * sum(r1[:symbol]) +
                   w2 * self._prefix(r2, symbol) +
                   w3 * self._prefix(r3, symbol) +
                   w4 * self._prefix(r4, symbol))
            return cum, freq, total

        cum = 0
        for s in range(256):
            freq = (self.W0 * self.zero[s] + self.W1 * r1[s] +
                    w2 * (r2.get(s, 0) if r2 else 0) +
                    w3 * (r3.get(s, 0) if r3 else 0) +
                    w4 * (r4.get(s, 0) if r4 else 0))
            if target < cum + freq:
                return s, cum, freq, total
            cum += freq
        return 255, cum - freq, freq, total

    @staticmethod
    def _bump(row, symbol):
        row[symbol] = row.get(symbol, 0) + 1

    @staticmethod
    def _trim(row, total):
        if total <= 2048:
            return total
        new_total = 0
        dead = []
        for symbol, count in row.items():
            count = (count + 1) >> 1
            if count:
                row[symbol] = count
                new_total += count
            else:
                dead.append(symbol)
        for symbol in dead:
            del row[symbol]
        return new_total

    def update(self, history, symbol):
        older = history & 255
        r1 = self.one[older]
        r1[symbol] += 17
        self.one_total[older] += 17
        if self.one_total[older] > 32768:
            total = 0
            for i, count in enumerate(r1):
                count = (count + 1) >> 1
                r1[i] = count
                total += count
            self.one_total[older] = total

        k2 = history & 65535
        r2 = self.two.get(k2)
        if r2 is None:
            r2 = {}
            self.two[k2] = r2
            self.two_total[k2] = 0
        self._bump(r2, symbol)
        self.two_total[k2] = self._trim(r2, self.two_total[k2] + 1)

        k3 = history & 0xFFFFFF
        r3 = self.three.get(k3)
        if r3 is None:
            r3 = {}
            self.three[k3] = r3
            self.three_total[k3] = 0
        self._bump(r3, symbol)
        self.three_total[k3] = self._trim(r3, self.three_total[k3] + 1)

        k4 = history & 0xFFFFFFFF
        r4 = self.four.get(k4)
        if r4 is None:
            r4 = {}
            self.four[k4] = r4
            self.four_total[k4] = 0
        self._bump(r4, symbol)
        self.four_total[k4] = self._trim(r4, self.four_total[k4] + 1)


def _emit(bits, bit, pending):
    bits.put(bit)
    while pending:
        bits.put(bit ^ 1)
        pending -= 1
    return pending


def compress(data: bytes) -> bytes:
    bits = _Bits()
    model = _Model()
    low, high, pending = 0, _FULL, 0
    history = 0
    for symbol in data:
        cum, freq, total = model.distribution(history, symbol=symbol)
        span = high - low + 1
        high = low + span * (cum + freq) // total - 1
        low += span * cum // total
        while True:
            if high < _HALF:
                pending = _emit(bits, 0, pending)
            elif low >= _HALF:
                pending = _emit(bits, 1, pending)
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
        model.update(history, symbol)
        history = ((history << 8) | symbol) & 0xFFFFFFFF
    pending += 1
    _emit(bits, 0 if low < _Q1 else 1, pending)
    bits.finish()
    return len(data).to_bytes(4, "big") + bytes(bits.data)


def decompress(blob: bytes) -> bytes:
    size = int.from_bytes(blob[:4], "big")
    bits = _Bits(blob[4:])
    model = _Model()
    low, high, value = 0, _FULL, 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    output = bytearray()
    history = 0
    for _ in range(size):
        span = high - low + 1
        total = model.distribution(history, target=0)[3]
        target = ((value - low + 1) * total - 1) // span
        symbol, cum, freq, total = model.distribution(history, target=target)
        high = low + span * (cum + freq) // total - 1
        low += span * cum // total
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
        output.append(symbol)
        model.update(history, symbol)
        history = ((history << 8) | symbol) & 0xFFFFFFFF
    return bytes(output)
