"""A small LZ parser with an adaptive PPM byte model.

The parser is deliberately independent of the coder: it chooses a sequence
of literals and copies by dynamic programming, while the coder gives literal
bytes a context-sensitive probability model.
"""
from bisect import bisect_left
import math


FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1
LITERAL_SCALE = 0.21
ESCAPE_COUNT = 3


class RangeCodec:
    def __init__(self, payload=b"", decoding=False):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0
        self.pos = 0
        self.low = 0
        self.high = FULL
        self.decoding = decoding
        self.value = 0
        if decoding:
            for _ in range(32):
                self.value = (self.value << 1) | self._read_bit(payload)
            self.payload = payload
        else:
            self.payload = b""

    def _write_bit(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def _read_bit(self, payload=None):
        if payload is None:
            payload = self.payload
        if self.pos >= len(payload) * 8:
            self.pos += 1
            return 0
        bit = (payload[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def put(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + span * (cumulative + frequency) // total - 1
        self.low = self.low + span * cumulative // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.npending += 1
                self.low -= Q1
                self.high -= Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def _emit(self, bit):
        self._write_bit(bit)
        while self.npending:
            self._write_bit(1 - bit)
            self.npending -= 1

    def begin(self):
        self.npending = 0

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def take(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + span * (cumulative + frequency) // total - 1
        self.low = self.low + span * cumulative // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.low -= Q1
                self.high -= Q1
                self.value -= Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._read_bit()

    def bit_put(self, bit):
        self.put(bit, 1, 2)

    def bit_take(self):
        target = self.target(2)
        bit = 1 if target else 0
        self.take(bit, 1, 2)
        return bit

    def finish(self):
        self.npending += 1
        self._emit(0 if self.low < Q1 else 1)
        while self.nbits:
            self._write_bit(0)
        return bytes(self.out)


class Context:
    __slots__ = ("keys", "counts", "total")

    def __init__(self, full=False):
        if full:
            self.keys = list(range(256))
            self.counts = {i: 1 for i in range(256)}
            self.total = 256
        else:
            self.keys = []
            self.counts = {}
            self.total = 0

    def add(self, symbol):
        old = self.counts.get(symbol)
        if old is None:
            k = bisect_left(self.keys, symbol)
            self.keys.insert(k, symbol)
            self.counts[symbol] = 1
            self.total += 1
        else:
            self.counts[symbol] = old + 1
            self.total += 1
        if self.total >= 4096:
            total = 0
            for k in self.keys:
                v = (self.counts[k] + 1) >> 1
                if v < 1:
                    v = 1
                self.counts[k] = v
                total += v
            self.total = total

    def put_symbol(self, rc, symbol, excluded=None):
        cumulative = 0
        for k in self.keys:
            if excluded is not None and k in excluded:
                continue
            v = self.counts[k]
            if k == symbol:
                total = self.total if excluded is None else sum(
                    self.counts[x] for x in self.keys if x not in excluded)
                rc.put(cumulative, v, total)
                return True
            cumulative += v
        return False

    def put_with_escape(self, rc, symbol, excluded):
        cumulative = 0
        available = sum(self.counts[k] for k in self.keys if k not in excluded)
        if available == 0:
            return None
        for k in self.keys:
            if k in excluded:
                continue
            v = self.counts[k]
            if k == symbol:
                rc.put(cumulative, v, available + ESCAPE_COUNT)
                return True
            cumulative += v
        rc.put(available, ESCAPE_COUNT, available + ESCAPE_COUNT)
        return False

    def take_with_escape(self, rc, excluded):
        available = sum(self.counts[k] for k in self.keys if k not in excluded)
        if available == 0:
            return -2
        target = rc.target(available + ESCAPE_COUNT)
        cumulative = 0
        for k in self.keys:
            if k in excluded:
                continue
            v = self.counts[k]
            if target < cumulative + v:
                rc.take(cumulative, v, available + ESCAPE_COUNT)
                return k
            cumulative += v
        rc.take(available, ESCAPE_COUNT, available + ESCAPE_COUNT)
        return -1

    def take_symbol(self, rc, excluded=None):
        total = self.total if excluded is None else sum(
            self.counts[k] for k in self.keys if k not in excluded)
        target = rc.target(total)
        cumulative = 0
        for k in self.keys:
            if excluded is not None and k in excluded:
                continue
            v = self.counts[k]
            if target < cumulative + v:
                rc.take(cumulative, v, total)
                return k
            cumulative += v
        return 0


class PPM:
    def __init__(self, maximum=3):
        self.maximum = maximum
        self.maps = [None] + [{} for _ in range(maximum)]
        self.zero = Context(full=True)
        self.history = 0
        self.have = 0

    def _key(self, order):
        return self.history & ((1 << (order * 8)) - 1)

    def encode(self, rc, symbol):
        excluded = set()
        for order in range(self.maximum, 0, -1):
            if self.have < order:
                continue
            context = self.maps[order].get(self._key(order))
            if context is not None:
                result = context.put_with_escape(rc, symbol, excluded)
                excluded.update(context.keys)
                if result is True:
                    self._update(symbol)
                    return
        self.zero.put_symbol(rc, symbol, excluded)
        self._update(symbol)

    def decode(self, rc):
        symbol = -1
        excluded = set()
        for order in range(self.maximum, 0, -1):
            if self.have < order:
                continue
            context = self.maps[order].get(self._key(order))
            if context is not None:
                symbol = context.take_with_escape(rc, excluded)
                excluded.update(context.keys)
                if symbol >= 0:
                    break
        if symbol < 0:
            symbol = self.zero.take_symbol(rc, excluded)
        self._update(symbol)
        return symbol

    def _update(self, symbol):
        for order in range(1, self.maximum + 1):
            if self.have >= order:
                key = self._key(order)
                context = self.maps[order].get(key)
                if context is None:
                    context = Context()
                    self.maps[order][key] = context
                context.add(symbol)
        self.zero.add(symbol)
        self.history = ((self.history << 8) | symbol) & 0xFFFFFF
        if self.have < self.maximum:
            self.have += 1


class TokenType:
    def __init__(self):
        self.literal = 1
        self.match = 1

    def put(self, rc, is_match):
        if is_match:
            rc.put(self.literal, self.match, self.literal + self.match)
            self.match += 1
        else:
            rc.put(0, self.literal, self.literal + self.match)
            self.literal += 1
        if self.literal + self.match >= 4096:
            self.literal = (self.literal + 1) >> 1
            self.match = (self.match + 1) >> 1

    def take(self, rc):
        total = self.literal + self.match
        target = rc.target(total)
        if target < self.literal:
            rc.take(0, self.literal, total)
            self.literal += 1
            answer = False
        else:
            rc.take(self.literal, self.match, total)
            self.match += 1
            answer = True
        if self.literal + self.match >= 4096:
            self.literal = (self.literal + 1) >> 1
            self.match = (self.match + 1) >> 1
        return answer


def _gamma_cost(value):
    return 2 * value.bit_length() - 1


def _write_gamma(rc, value):
    width = value.bit_length() - 1
    for _ in range(width):
        rc.bit_put(0)
    rc.bit_put(1)
    for shift in range(width - 1, -1, -1):
        rc.bit_put((value >> shift) & 1)


def _read_gamma(rc):
    width = 0
    while rc.bit_take() == 0:
        width += 1
    value = 1 << width
    for shift in range(width - 1, -1, -1):
        value |= rc.bit_take() << shift
    return value


def _parse(data):
    n = len(data)
    if n == 0:
        return []
    counts = [1] * 256
    for symbol in data:
        counts[symbol] += 1
    total = n + 256
    # The PPM coder is substantially cheaper than a raw byte.  This factor
    # keeps the parse's planning metric in the same range as its emitted
    # literal costs, without making the parser depend on coder state.
    literal_cost = [LITERAL_SCALE * (1.0 - math.log2(c / total)) for c in counts]

    positions = {}
    for i in range(max(0, n - 3)):
        key = data[i:i + 4]
        positions.setdefault(key, []).append(i)

    inf = float("inf")
    costs = [inf] * (n + 1)
    choice_kind = [0] * n
    choice_arg = [0] * n
    choice_len = [1] * n
    costs[n] = 0.0
    max_match = 1023
    max_candidates = 32

    for i in range(n - 1, -1, -1):
        candidate = literal_cost[data[i]] + costs[i + 1]
        if candidate < costs[i]:
            costs[i] = candidate
            choice_kind[i] = 0
            choice_arg[i] = data[i]
            choice_len[i] = 1
        if i + 4 > n:
            continue
        occurrences = positions.get(data[i:i + 4])
        if not occurrences:
            continue
        stop = bisect_left(occurrences, i)
        first = max(0, stop - max_candidates)
        for p in range(stop - 1, first - 1, -1):
            previous = occurrences[p]
            distance = i - previous
            if distance > 65535:
                break
            limit = min(max_match, n - i)
            length = 4
            while length < limit:
                source = previous + length
                expected = data[source] if source < i else data[i + length - distance]
                if data[i + length] != expected:
                    break
                length += 1
            if length < 4:
                continue
            fixed = 1.0 + _gamma_cost(distance)
            for match_length in range(4, length + 1):
                end = i + match_length
                candidate = fixed + _gamma_cost(match_length - 3) + costs[end]
                if candidate < costs[i]:
                    costs[i] = candidate
                    choice_kind[i] = 1
                    choice_arg[i] = distance
                    choice_len[i] = match_length

    tokens = []
    i = 0
    while i < n:
        if choice_kind[i]:
            distance = choice_arg[i]
            length = choice_len[i]
            tokens.append((1, distance, length))
            i += length
        else:
            tokens.append((0, choice_arg[i], 1))
            i += 1
    return tokens


def compress(data: bytes) -> bytes:
    data = bytes(data)
    tokens = _parse(data)
    rc = RangeCodec()
    rc.begin()
    types = TokenType()
    model = PPM(3)
    source_pos = 0
    for kind, argument, length in tokens:
        types.put(rc, kind == 1)
        if kind == 0:
            model.encode(rc, argument)
            source_pos += 1
        else:
            _write_gamma(rc, argument)
            _write_gamma(rc, length - 3)
            for symbol in data[source_pos:source_pos + length]:
                model._update(symbol)
            source_pos += length
    return len(data).to_bytes(4, "big") + rc.finish()


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    rc = RangeCodec(blob[4:], decoding=True)
    types = TokenType()
    model = PPM(3)
    output = bytearray()
    while len(output) < n:
        if types.take(rc):
            distance = _read_gamma(rc)
            length = _read_gamma(rc) + 3
            start = len(output) - distance
            for _ in range(length):
                symbol = output[start]
                start += 1
                output.append(symbol)
                model._update(symbol)
        else:
            symbol = model.decode(rc)
            output.append(symbol)
    return bytes(output)
