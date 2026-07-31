"""A small adaptive order-one arithmetic coder for the text slice."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
Q1 = 0x40000000
Q3 = 0xC0000000


class _Bits:
    def __init__(self, raw=b""):
        self.buf = bytearray(raw)
        self.acc = 0
        self.n = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.n += 1
        if self.n == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.n = 0

    def finish(self):
        while self.n:
            self.put(0)

    def get(self):
        if self.pos >= len(self.buf) * 8:
            return 0
        v = self.buf[self.pos >> 3]
        bit = (v >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class _Contexts:
    def __init__(self):
        # One byte of history.  The unit count of one keeps every symbol
        # legal, while the active count measures how concentrated a context
        # has become for the live rescaling rule.
        self.freq = [[1] * 256 for _ in range(256)]
        self.total = [256] * 256
        self.active = [0] * 256

    def row(self, context):
        return self.freq[context]

    def update(self, context, symbol):
        row = self.freq[context]
        if row[symbol] == 1:
            self.active[context] += 1
        row[symbol] += 17
        total = self.total[context] + 17

        # The cap is intentionally a live property of this context.  A
        # concentrated context is allowed to learn quickly with a smaller
        # effective memory; a diverse one gets more mass before forgetting.
        cap = 1024 + 640 * self.active[context]
        if cap > 32768:
            cap = 32768
        if total > cap:
            new_total = 0
            new_active = 0
            for i in range(256):
                v = (row[i] + 1) >> 1
                row[i] = v
                new_total += v
                if v > 1:
                    new_active += 1
            self.total[context] = new_total
            self.active[context] = new_active
        else:
            self.total[context] = total


def _emit(out, pending, bit):
    out.put(bit)
    while pending:
        out.put(1 - bit)
        pending -= 1
    return pending


def compress(data: bytes) -> bytes:
    out = _Bits()
    model = _Contexts()
    low, high, pending = 0, FULL, 0
    context = 0

    for symbol in data:
        row = model.row(context)
        cum = 0
        for i in range(symbol):
            cum += row[i]
        freq = row[symbol]
        total = model.total[context]
        span = high - low + 1
        high = low + (span * (cum + freq) // total) - 1
        low = low + (span * cum // total)
        while True:
            if high < HALF:
                pending = _emit(out, pending, 0)
            elif low >= HALF:
                pending = _emit(out, pending, 1)
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
        model.update(context, symbol)
        context = symbol

    pending += 1
    _emit(out, pending, 0 if low < Q1 else 1)
    out.finish()
    return len(data).to_bytes(4, "big") + bytes(out.buf)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    bits = _Bits(blob[4:])
    model = _Contexts()
    low, high = 0, FULL
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()

    out = bytearray()
    context = 0
    for _ in range(n):
        total = model.total[context]
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        row = model.row(context)
        cum = 0
        symbol = 0
        for i in range(256):
            f = row[i]
            if target < cum + f:
                symbol = i
                freq = f
                break
            cum += f
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
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.get()
        out.append(symbol)
        model.update(context, symbol)
        context = symbol
    return bytes(out)
