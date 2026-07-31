"""Deterministic lossless LZ parser with future-cost token selection."""


MIN_MATCH = 4
MAX_MATCH = 258
MAX_DISTANCE = 65535
CANDIDATES = 512


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def put(self, value, width):
        if width:
            self.acc = (self.acc << width) | value
            self.nbits += width
            while self.nbits >= 8:
                self.nbits -= 8
                self.out.append((self.acc >> self.nbits) & 255)
                if self.nbits:
                    self.acc &= (1 << self.nbits) - 1
                else:
                    self.acc = 0

    def finish(self):
        if self.nbits:
            self.out.append((self.acc << (8 - self.nbits)) & 255)
        return bytes(self.out)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self, width):
        value = 0
        for _ in range(width):
            if self.pos >> 3 < len(self.data):
                value = (value << 1) | ((self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1)
            else:
                value <<= 1
            self.pos += 1
        return value


def _put_gamma(bits, value):
    width = value.bit_length() - 1
    bits.put(0, width)
    bits.put(value, width + 1)


def _gamma_bits(value):
    return 2 * value.bit_length() - 1


def _get_gamma(bits):
    zeros = 0
    while bits.get(1) == 0:
        zeros += 1
    value = 1
    if zeros:
        value = (1 << zeros) | bits.get(zeros)
    return value


def _matches(data):
    """Find a bounded best prior match at every source position."""
    n = len(data)
    prior = {}
    best = [(0, 0)] * n
    for i in range(n):
        if i + MIN_MATCH <= n:
            key = data[i:i + MIN_MATCH]
            candidates = prior.get(key, ())
            longest = 0
            distance = 0
            limit = min(MAX_MATCH, n - i)
            for p in reversed(candidates):
                if i - p > MAX_DISTANCE:
                    break
                length = MIN_MATCH
                while length < limit and data[p + length] == data[i + length]:
                    length += 1
                if length > longest:
                    longest = length
                    distance = i - p
                    if length == limit:
                        break
            if longest >= MIN_MATCH:
                best[i] = (longest, distance)
            positions = prior.setdefault(key, [])
            positions.append(i)
            if len(positions) > CANDIDATES:
                del positions[0]
    return best


def _parse(data, best):
    """Minimize exact token-bit cost, including the cost after each choice."""
    n = len(data)
    future = [0] * (n + 1)
    choice = [0] * n
    for i in range(n - 1, -1, -1):
        selected = 9 + future[i + 1]
        selected_length = 0
        longest, distance = best[i]
        if longest:
            match_cost = 1 + 16
            for length in range(MIN_MATCH, longest + 1):
                cost = match_cost + _gamma_bits(length - 3) + future[i + length]
                if cost < selected:
                    selected = cost
                    selected_length = length
        future[i] = selected
        choice[i] = selected_length
    return choice


def compress(data: bytes) -> bytes:
    n = len(data)
    best = _matches(data)
    choice = _parse(data, best)
    bits = BitWriter()
    i = 0
    while i < n:
        length = choice[i]
        if length:
            bits.put(1, 1)
            bits.put(best[i][1] - 1, 16)
            _put_gamma(bits, length - 3)
            i += length
        else:
            bits.put(0, 1)
            bits.put(data[i], 8)
            i += 1
    return n.to_bytes(4, "big") + bits.finish()


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    bits = BitReader(blob[4:])
    out = bytearray()
    while len(out) < n:
        if bits.get(1) == 0:
            out.append(bits.get(8))
        else:
            distance = bits.get(16) + 1
            length = _get_gamma(bits) + 3
            start = len(out) - distance
            for _ in range(length):
                out.append(out[start])
                start += 1
    return bytes(out)
