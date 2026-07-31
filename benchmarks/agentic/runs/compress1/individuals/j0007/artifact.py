"""Order-two arithmetic coding with an order-one probabilistic backoff."""

TOP = 0xFFFFFFFF
MID = 0x80000000
LO_Q = 0x40000000
HI_Q = 0xC0000000
BACKOFF = 8


class _Stream:
    def __init__(self, raw=b""):
        self.buf = bytearray(raw)
        self.acc = 0
        self.used = 0
        self.at = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.used += 1
        if self.used == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.used = 0

    def read(self):
        if self.at >= len(self.buf) * 8:
            return 0
        b = self.buf[self.at >> 3]
        bit = (b >> (7 - (self.at & 7))) & 1
        self.at += 1
        return bit

    def close(self):
        while self.used:
            self.write(0)


class _Model:
    def __init__(self):
        self.one = [[1] * 256 for _ in range(256)]
        self.one_total = [256] * 256
        self.one_live = [0] * 256
        self.two = {}
        self.two_total = {}

    def _two_row(self, context):
        row = self.two.get(context)
        if row is None:
            row = [0] * 256
            self.two[context] = row
            self.two_total[context] = 0
        return row

    def distribution(self, context, symbol=None, target=None):
        older = context & 255
        r1 = self.one[older]
        t1 = self.one_total[older]
        r2 = self._two_row(context)
        n2 = self.two_total[context]
        total = t1 * (n2 + BACKOFF)
        if symbol is not None:
            c2 = sum(r2[:symbol])
            c1 = sum(r1[:symbol])
            f = r2[symbol] * t1 + BACKOFF * r1[symbol]
            return c2 * t1 + BACKOFF * c1, f, total
        c2 = 0
        c1 = 0
        for s in range(256):
            f = r2[s] * t1 + BACKOFF * r1[s]
            if target < c2 * t1 + BACKOFF * c1 + f:
                return s, c2 * t1 + BACKOFF * c1, f, total
            c2 += r2[s]
            c1 += r1[s]
        s = 255
        return s, c2 * t1 + BACKOFF * c1 - (r2[s] * t1 + BACKOFF * r1[s]), r2[s] * t1 + BACKOFF * r1[s], total

    def update(self, context, symbol):
        older = context & 255
        r1 = self.one[older]
        if r1[symbol] == 1:
            self.one_live[older] += 1
        r1[symbol] += 17
        total = self.one_total[older] + 17
        cap = 1024 + 640 * self.one_live[older]
        if cap > 32768:
            cap = 32768
        if total > cap:
            new_total = 0
            live = 0
            for i in range(256):
                v = (r1[i] + 1) >> 1
                r1[i] = v
                new_total += v
                if v > 1:
                    live += 1
            self.one_total[older] = new_total
            self.one_live[older] = live
        else:
            self.one_total[older] = total

        r2 = self._two_row(context)
        r2[symbol] += 1
        n2 = self.two_total[context] + 1
        if n2 > 2048:
            n2 = 0
            for i in range(256):
                r2[i] >>= 1
                n2 += r2[i]
        self.two_total[context] = n2


def _put(out, bit, pending):
    out.write(bit)
    while pending:
        out.write(bit ^ 1)
        pending -= 1
    return pending


def compress(data: bytes) -> bytes:
    out = _Stream()
    model = _Model()
    low, high, pending = 0, TOP, 0
    context = 0
    for symbol in data:
        cum, freq, total = model.distribution(context, symbol=symbol)
        span = high - low + 1
        high = low + span * (cum + freq) // total - 1
        low += span * cum // total
        while True:
            if high < MID:
                pending = _put(out, 0, pending)
            elif low >= MID:
                pending = _put(out, 1, pending)
                low -= MID
                high -= MID
            elif low >= LO_Q and high < HI_Q:
                pending += 1
                low -= LO_Q
                high -= LO_Q
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        model.update(context, symbol)
        context = ((context & 255) << 8) | symbol
    pending += 1
    _put(out, 0 if low < LO_Q else 1, pending)
    out.close()
    return len(data).to_bytes(4, "big") + bytes(out.buf)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    bits = _Stream(blob[4:])
    model = _Model()
    low, high = 0, TOP
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.read()
    out = bytearray()
    context = 0
    for _ in range(n):
        span = high - low + 1
        total = model.distribution(context, target=0)[3]
        target = ((value - low + 1) * total - 1) // span
        symbol, cum, freq, total = model.distribution(context, target=target)
        high = low + span * (cum + freq) // total - 1
        low += span * cum // total
        while True:
            if high < MID:
                pass
            elif low >= MID:
                low -= MID
                high -= MID
                value -= MID
            elif low >= LO_Q and high < HI_Q:
                low -= LO_Q
                high -= LO_Q
                value -= LO_Q
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.read()
        out.append(symbol)
        model.update(context, symbol)
        context = ((context & 255) << 8) | symbol
    return bytes(out)
