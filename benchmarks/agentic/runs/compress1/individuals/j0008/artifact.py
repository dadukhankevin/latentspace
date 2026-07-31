"""Contextual lossless byte compressor for the compress benchmark."""

FULL = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREE_QUARTER = HALF + QUARTER
MAX_ORDER = 4


class _BitWriter:
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


class _BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        q, r = divmod(self.pos, 8)
        self.pos += 1
        return (self.data[q] >> (7 - r)) & 1


def _limit(total, dominant):
    # A concentrated distribution has little useful tail history.  A
    # dispersed distribution gets a limit proportional to its tail mass.
    return min(65536, 64 + total - dominant)


def _halve(mapping):
    total = 0
    for symbol in tuple(mapping):
        value = (mapping[symbol] + 1) // 2
        mapping[symbol] = value
        total += value
    return total


class _PPM:
    def __init__(self):
        self.zero = [1] * 256
        self.zero_total = 256
        self.contexts = [{} for _ in range(MAX_ORDER)]

    def _entry(self, order, key, create):
        table = self.contexts[order - 1]
        entry = table.get(key)
        if entry is None and create:
            entry = [{}, 0]
            table[key] = entry
        return entry

    def _update_context(self, order, key, symbol):
        entry = self._entry(order, key, True)
        counts = entry[0]
        counts[symbol] = counts.get(symbol, 0) + 1
        total = entry[1] + 1
        entry[1] = total
        if total > _limit(total, max(counts.values())):
            entry[1] = _halve(counts)

    def update(self, history, symbol):
        self.zero[symbol] += 1
        self.zero_total += 1
        self.zero_rescale()

        for order in range(1, min(MAX_ORDER, len(history)) + 1):
            key = 0
            for item in history[-order:]:
                key = (key << 8) | item
            self._update_context(order, key, symbol)

    def zero_rescale(self):
        dominant = max(self.zero)
        if self.zero_total > _limit(self.zero_total, dominant):
            total = 0
            for i, value in enumerate(self.zero):
                value = (value + 1) // 2
                self.zero[i] = value
                total += value
            self.zero_total = total


def _new_model():
    return _PPM()


def _context_keys(history):
    keys = []
    for order in range(min(MAX_ORDER, len(history)), 0, -1):
        key = 0
        for item in history[-order:]:
            key = (key << 8) | item
        keys.append((order, key))
    return keys


def _encode_symbol(coder, model, history, symbol):
    excluded = set()
    for order, key in _context_keys(history):
        entry = model._entry(order, key, False)
        if entry is None or not entry[0]:
            continue
        counts = entry[0]
        available = 0
        for value, frequency in counts.items():
            if value not in excluded:
                available += frequency
        if not available:
            excluded.update(counts)
            continue
        total = available + 1
        cumulative = 0
        found = False
        for value, frequency in counts.items():
            if value in excluded:
                continue
            if value == symbol:
                coder.encode(cumulative, frequency, total)
                found = True
                break
            cumulative += frequency
        if found:
            return
        coder.encode(available, 1, total)
        excluded.update(counts)

    cumulative = 0
    available = model.zero_total
    for value in excluded:
        available -= model.zero[value]
    for value in range(symbol):
        if value not in excluded:
            cumulative += model.zero[value]
    coder.encode(cumulative, model.zero[symbol], available)


def _decode_symbol(coder, model, history):
    excluded = set()
    for order, key in _context_keys(history):
        entry = model._entry(order, key, False)
        if entry is None or not entry[0]:
            continue
        counts = entry[0]
        available = 0
        for value, frequency in counts.items():
            if value not in excluded:
                available += frequency
        if not available:
            excluded.update(counts)
            continue
        total = available + 1
        target = coder.target(total)
        cumulative = 0
        chosen = None
        for value, frequency in counts.items():
            if value in excluded:
                continue
            if target < cumulative + frequency:
                chosen = value
                coder.remove(cumulative, frequency, total)
                break
            cumulative += frequency
        if chosen is not None:
            return chosen
        coder.remove(available, 1, total)
        excluded.update(counts)

    available = model.zero_total
    for value in excluded:
        available -= model.zero[value]
    target = coder.target(available)
    cumulative = 0
    for value, frequency in enumerate(model.zero):
        if value in excluded:
            continue
        if target < cumulative + frequency:
            coder.remove(cumulative, frequency, available)
            return value
        cumulative += frequency
    return 255


class _ArithmeticEncoder:
    def __init__(self):
        self.bits = _BitWriter()
        self.low = 0
        self.high = FULL
        self.pending = 0

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(bit ^ 1)
            self.pending -= 1

    def encode(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency)) // total - 1
        self.low = self.low + (span * cumulative) // total
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

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        self.bits.used = self.bits.used
        return self.bits.finish()


class _ArithmeticDecoder:
    def __init__(self, data):
        self.bits = _BitReader(data)
        self.low = 0
        self.high = FULL
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.get()

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def remove(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency)) // total - 1
        self.low = self.low + (span * cumulative) // total
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


def compress(data: bytes) -> bytes:
    model = _new_model()
    coder = _ArithmeticEncoder()
    history = []
    for symbol in data:
        _encode_symbol(coder, model, history, symbol)
        model.update(history, symbol)
        history.append(symbol)
        if len(history) > MAX_ORDER:
            del history[0]
    return len(data).to_bytes(4, "big") + coder.finish()


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    model = _new_model()
    coder = _ArithmeticDecoder(blob[4:])
    history = []
    out = bytearray()
    for _ in range(n):
        symbol = _decode_symbol(coder, model, history)
        out.append(symbol)
        model.update(history, symbol)
        history.append(symbol)
        if len(history) > MAX_ORDER:
            del history[0]
    return bytes(out)
