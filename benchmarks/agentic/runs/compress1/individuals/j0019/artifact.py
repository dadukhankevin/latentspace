"""Adaptive order-2 PPM coding for arbitrary byte strings."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = HALF + QUARTER
LIMIT = 60000


class _BitsOut:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.bits = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.bits += 1
        if self.bits == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.bits = 0

    def finish(self):
        if self.bits:
            self.acc <<= 8 - self.bits
            self.buf.append(self.acc)


class _BitsIn:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        p = self.pos
        self.pos += 1
        return (self.data[p >> 3] >> (7 - (p & 7))) & 1


class _Arithmetic:
    def __init__(self, source=None):
        self.low = 0
        self.high = FULL
        self.pending = 0
        self.source = source
        self.value = 0
        if source is not None:
            for _ in range(32):
                self.value = (self.value << 1) | source.get()

    def _emit(self, bit, sink):
        sink.put(bit)
        while self.pending:
            sink.put(bit ^ 1)
            self.pending -= 1

    def encode(self, cum, freq, total, sink):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cum + freq)) // total - 1
        self.low = self.low + (span * cum) // total
        while True:
            if self.high < HALF:
                self._emit(0, sink)
            elif self.low >= HALF:
                self._emit(1, sink)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def decode(self, cum, freq, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cum + freq)) // total - 1
        self.low = self.low + (span * cum) // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.source.get()


class _Context:
    def __init__(self):
        self.counts = {}
        self.total = 0

    def escape_frequency(self):
        return max(1, len(self.counts))

    def event_for(self, symbol):
        """Return cumulative, frequency, and total for symbol or escape."""
        cumulative = 0
        for old, count in self.counts.items():
            if old == symbol:
                return cumulative, count, self.total + self.escape_frequency()
            cumulative += count
        esc = self.escape_frequency()
        return self.total, esc, self.total + esc

    def symbol_for(self, target):
        """Return (symbol, cumulative, frequency, total), or escape symbol."""
        esc = self.escape_frequency()
        total = self.total + esc
        if target >= self.total:
            return None, self.total, esc, total
        cumulative = 0
        for symbol, count in self.counts.items():
            if target < cumulative + count:
                return symbol, cumulative, count, total
            cumulative += count
        return None, self.total, esc, total

    def add(self, symbol):
        old = self.counts.get(symbol)
        if old is None:
            self.counts[symbol] = 1
        else:
            self.counts[symbol] = old + 1
        self.total += 1
        if self.total > LIMIT:
            total = 0
            for key, count in self.counts.items():
                reduced = (count + 1) >> 1
                if reduced == 0:
                    reduced = 1
                self.counts[key] = reduced
                total += reduced
            self.total = total


def _new_model():
    return {}, {}, {}, _Context()


def _contexts_for_encode(d3, d2, d1, d0, prev1, prev2, prev3, symbol):
    if prev3 is not None:
        ctx = d3.get((prev3 << 16) | (prev2 << 8) | prev1)
        if ctx is not None and ctx.counts:
            yield ctx, symbol
    if prev2 is not None:
        ctx = d2.get((prev2 << 8) | prev1)
        if ctx is not None and ctx.counts:
            yield ctx, symbol
    if prev1 is not None:
        ctx = d1.get(prev1)
        if ctx is not None and ctx.counts:
            yield ctx, symbol
    if d0.counts:
        yield d0, symbol


def compress(data: bytes) -> bytes:
    n = len(data)
    if n == 0:
        return b"\x00\x00\x00\x00"
    d3, d2, d1, d0 = _new_model()
    sink = _BitsOut()
    coder = _Arithmetic()
    prev1 = None
    prev2 = None
    prev3 = None
    for symbol in data:
        encoded = False
        for ctx, wanted in _contexts_for_encode(d3, d2, d1, d0, prev1, prev2, prev3, symbol):
            cum, freq, total = ctx.event_for(wanted)
            coder.encode(cum, freq, total, sink)
            if wanted in ctx.counts:
                encoded = True
                break
        if not encoded:
            coder.encode(symbol, 1, 256, sink)

        if prev3 is not None:
            key = (prev3 << 16) | (prev2 << 8) | prev1
            ctx = d3.get(key)
            if ctx is None:
                ctx = _Context()
                d3[key] = ctx
            ctx.add(symbol)
        if prev2 is not None:
            key = (prev2 << 8) | prev1
            ctx = d2.get(key)
            if ctx is None:
                ctx = _Context()
                d2[key] = ctx
            ctx.add(symbol)
        if prev1 is not None:
            ctx = d1.get(prev1)
            if ctx is None:
                ctx = _Context()
                d1[prev1] = ctx
            ctx.add(symbol)
        d0.add(symbol)
        prev3, prev2, prev1 = prev2, prev1, symbol

    coder.pending += 1
    coder._emit(0 if coder.low < QUARTER else 1, sink)
    sink.finish()
    return n.to_bytes(4, "big") + bytes(sink.buf)


def decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        return b""
    n = int.from_bytes(blob[:4], "big")
    if n == 0:
        return b""
    d3, d2, d1, d0 = _new_model()
    source = _BitsIn(blob[4:])
    coder = _Arithmetic(source)
    out = bytearray()
    prev1 = None
    prev2 = None
    prev3 = None
    for _ in range(n):
        symbol = None
        for ctx, _ in _contexts_for_encode(d3, d2, d1, d0, prev1, prev2, prev3, 0):
            total = ctx.total + ctx.escape_frequency()
            target = coder.target(total)
            chosen, cum, freq, _ = ctx.symbol_for(target)
            coder.decode(cum, freq, total)
            if chosen is not None:
                symbol = chosen
                break
        if symbol is None:
            total = 256
            target = coder.target(total)
            symbol = target
            coder.decode(symbol, 1, total)

        out.append(symbol)
        if prev3 is not None:
            key = (prev3 << 16) | (prev2 << 8) | prev1
            ctx = d3.get(key)
            if ctx is None:
                ctx = _Context()
                d3[key] = ctx
            ctx.add(symbol)
        if prev2 is not None:
            key = (prev2 << 8) | prev1
            ctx = d2.get(key)
            if ctx is None:
                ctx = _Context()
                d2[key] = ctx
            ctx.add(symbol)
        if prev1 is not None:
            ctx = d1.get(prev1)
            if ctx is None:
                ctx = _Context()
                d1[prev1] = ctx
            ctx.add(symbol)
        d0.add(symbol)
        prev3, prev2, prev1 = prev2, prev1, symbol
    return bytes(out)
