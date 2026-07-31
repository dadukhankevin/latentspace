"""A small adaptive sparse PPM coder with a synchronized escape prior."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = 0xC0000000

# The two constants are deliberately conservative.  Contexts that recur
# rapidly keep their escape cheap; a context that has gone cold gets a little
# more room for an unobserved continuation, but never dominates its symbols.
AGE_STEP = 32768
ESCAPE_CAP = 20
COUNT_LIMIT = 2048
ROOT_LIMIT = 60000


class _BitsOut:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.nbits = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | int(bit)
        self.nbits += 1
        if self.nbits == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def finish(self):
        if self.nbits:
            self.buf.append(self.acc << (8 - self.nbits))
        return bytes(self.buf)


class _BitsIn:
    def __init__(self, raw):
        self.raw = raw
        self.at = 0

    def get(self):
        if self.at >= len(self.raw) * 8:
            return 0
        p = self.at
        self.at += 1
        return (self.raw[p >> 3] >> (7 - (p & 7))) & 1


class _Range:
    """The interval coder uses the same integer recurrence in both modes."""

    def __init__(self, writing, payload=b""):
        self.writing = writing
        self.low = 0
        self.high = FULL
        self.pending = 0
        if writing:
            self.bits = _BitsOut()
        else:
            self.bits = _BitsIn(payload)
            self.value = 0
            for _ in range(32):
                self.value = (self.value << 1) | self.bits.get()

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(1 - bit)
            self.pending -= 1

    def encode(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
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

    def choose(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def decode_interval(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
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
            self.value = (self.value << 1) | self.bits.get()

    def close(self):
        if self.writing:
            self.pending += 1
            self._emit(0 if self.low < QUARTER else 1)
            return self.bits.finish()
        return b""


class _PPM:
    def __init__(self):
        # A root prior keeps every byte available from the start.
        self.root = [1] * 256
        # Each value is [continuation-count dictionary, last position].
        self.tables = [{}, {}, {}, {}, {}]
        self.history = []
        self.pos = 0

    def _escape(self, state):
        counts, previous = state
        age = self.pos - previous if previous >= 0 else 0
        e = len(counts) + age // AGE_STEP
        if e < 1:
            e = 1
        return ESCAPE_CAP if e > ESCAPE_CAP else e

    @staticmethod
    def _add_excluded(excluded, counts):
        for symbol in counts:
            excluded.add(symbol)

    def _state(self, order):
        key = tuple(self.history[-order:])
        return self.tables[order].get(key), key

    def _root_encode(self, symbol, excluded, coder):
        cumulative = 0
        total = 0
        for value in self.root:
            total += value
        for s in range(symbol):
            if s not in excluded:
                cumulative += self.root[s]
        frequency = self.root[symbol]
        coder.encode(cumulative, frequency, total - sum(self.root[s] for s in excluded))

    def _root_decode(self, excluded, coder):
        total = sum(self.root[s] for s in range(256) if s not in excluded)
        target = coder.choose(total)
        cumulative = 0
        for symbol, frequency in enumerate(self.root):
            if symbol in excluded:
                continue
            if target < cumulative + frequency:
                coder.decode_interval(cumulative, frequency, total)
                return symbol
            cumulative += frequency
        raise ValueError("root model desynchronized")

    def _advance(self, symbol):
        self.root[symbol] += 1
        root_total = sum(self.root)
        if root_total > ROOT_LIMIT:
            self.root = [(v + 1) // 2 for v in self.root]

        max_order = 4 if len(self.history) >= 4 else len(self.history)
        for order in range(1, max_order + 1):
            key = tuple(self.history[-order:])
            state = self.tables[order].get(key)
            if state is None:
                state = [{}, -1]
                self.tables[order][key] = state
            counts = state[0]
            counts[symbol] = counts.get(symbol, 0) + 1
            if sum(counts.values()) > COUNT_LIMIT:
                for s in list(counts):
                    counts[s] = (counts[s] + 1) // 2
            state[1] = self.pos
        self.history.append(symbol)
        if len(self.history) > 4:
            del self.history[0]
        self.pos += 1

    def encode_symbol(self, symbol, coder):
        excluded = set()
        selected = False
        for order in range(4, 0, -1):
            state, _ = self._state(order)
            if state is None:
                continue
            counts = state[0]
            available = []
            total = 0
            for s, frequency in counts.items():
                if s not in excluded:
                    available.append((s, frequency))
                    total += frequency
            if not available:
                continue
            escape = self._escape(state) if len(counts) < 256 else 0
            if symbol in counts and symbol not in excluded:
                cumulative = 0
                for s, frequency in available:
                    if s == symbol:
                        coder.encode(cumulative, frequency, total + escape)
                        selected = True
                        break
                    cumulative += frequency
                if selected:
                    break
            if escape == 0:
                self._add_excluded(excluded, counts)
                continue
            coder.encode(total, escape, total + escape)
            self._add_excluded(excluded, counts)

        if not selected:
            self._root_encode(symbol, excluded, coder)
        self._advance(symbol)

    def decode_symbol(self, coder):
        excluded = set()
        for order in range(4, 0, -1):
            state, _ = self._state(order)
            if state is None:
                continue
            counts = state[0]
            available = []
            total = 0
            for s, frequency in counts.items():
                if s not in excluded:
                    available.append((s, frequency))
                    total += frequency
            if not available:
                continue
            escape = self._escape(state) if len(counts) < 256 else 0
            if escape == 0:
                self._add_excluded(excluded, counts)
                continue
            full = total + escape
            target = coder.choose(full)
            if target < total:
                cumulative = 0
                for symbol, frequency in available:
                    if target < cumulative + frequency:
                        coder.decode_interval(cumulative, frequency, full)
                        self._advance(symbol)
                        return symbol
                    cumulative += frequency
                raise ValueError("context model desynchronized")
            coder.decode_interval(total, escape, full)
            self._add_excluded(excluded, counts)

        symbol = self._root_decode(excluded, coder)
        self._advance(symbol)
        return symbol


def compress(data: bytes) -> bytes:
    raw = bytes(data)
    coder = _Range(True)
    model = _PPM()
    for symbol in raw:
        model.encode_symbol(symbol, coder)
    return len(raw).to_bytes(4, "big") + coder.close()


def decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        raise ValueError("truncated header")
    length = int.from_bytes(blob[:4], "big")
    coder = _Range(False, blob[4:])
    model = _PPM()
    output = bytearray()
    for _ in range(length):
        output.append(model.decode_symbol(coder))
    return bytes(output)
