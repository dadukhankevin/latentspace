from array import array


_TOP = 0xFFFFFFFF
_MID = 0x80000000
_LOW_QUARTER = 0x40000000
_HIGH_QUARTER = 0xC0000000
_STEP = 17
_PAIR_LIMIT = 8192
_BYTE_LIMIT = 8192


class _BitStream:
    def __init__(self, payload=b""):
        self.data = bytearray(payload)
        self.accumulator = 0
        self.bits = 0
        self.cursor = 0

    def write(self, bit):
        self.accumulator = (self.accumulator << 1) | bit
        self.bits += 1
        if self.bits == 8:
            self.data.append(self.accumulator)
            self.accumulator = 0
            self.bits = 0

    def pad(self):
        while self.bits:
            self.write(0)

    def read(self):
        if self.cursor >= len(self.data) * 8:
            return 0
        byte = self.data[self.cursor >> 3]
        bit = (byte >> (7 - (self.cursor & 7))) & 1
        self.cursor += 1
        return bit


class _Model:
    def __init__(self):
        self.byte_counts = [[1] * 256 for _ in range(256)]
        self.byte_totals = [256] * 256
        self.pair_counts = array("H", [1]) * (65536 * 256)
        self.pair_totals = [256] * 65536
        self.pair_seen = bytearray(65536)

    def distribution(self, pair, previous):
        if self.pair_seen[pair] >= 4:
            return self.pair_counts, pair << 8, self.pair_totals[pair]
        return self.byte_counts[previous], 0, self.byte_totals[previous]

    def observe(self, pair, previous, symbol):
        row = self.byte_counts[previous]
        row[symbol] += _STEP
        total = self.byte_totals[previous] + _STEP
        if total > _BYTE_LIMIT:
            total = 0
            for i in range(256):
                value = (row[i] + 1) >> 1
                if value < 1:
                    value = 1
                row[i] = value
                total += value
        self.byte_totals[previous] = total

        start = pair << 8
        index = start + symbol
        self.pair_counts[index] += _STEP
        ptotal = self.pair_totals[pair] + _STEP
        if ptotal > _PAIR_LIMIT:
            ptotal = 0
            for i in range(start, start + 256):
                value = (self.pair_counts[i] + 1) >> 1
                if value < 1:
                    value = 1
                self.pair_counts[i] = value
                ptotal += value
        self.pair_totals[pair] = ptotal
        if self.pair_seen[pair] != 255:
            self.pair_seen[pair] += 1


def _output_bit(stream, waiting, bit):
    stream.write(bit)
    while waiting:
        stream.write(1 - bit)
        waiting -= 1
    return waiting


def compress(data: bytes) -> bytes:
    stream = _BitStream()
    model = _Model()
    low = 0
    high = _TOP
    waiting = 0
    previous = 0
    older = 0

    for symbol in data:
        pair = (older << 8) | previous
        row, start, total = model.distribution(pair, previous)
        cumulative = 0
        for candidate in range(symbol):
            cumulative += row[start + candidate]
        frequency = row[start + symbol]
        span = high - low + 1
        high = low + (span * (cumulative + frequency) // total) - 1
        low = low + (span * cumulative // total)
        while True:
            if high < _MID:
                waiting = _output_bit(stream, waiting, 0)
            elif low >= _MID:
                waiting = _output_bit(stream, waiting, 1)
                low -= _MID
                high -= _MID
            elif low >= _LOW_QUARTER and high < _HIGH_QUARTER:
                waiting += 1
                low -= _LOW_QUARTER
                high -= _LOW_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        model.observe(pair, previous, symbol)
        older, previous = previous, symbol

    waiting += 1
    _output_bit(stream, waiting, 0 if low < _LOW_QUARTER else 1)
    stream.pad()
    return len(data).to_bytes(4, "big") + bytes(stream.data)


def decompress(blob: bytes) -> bytes:
    length = int.from_bytes(blob[:4], "big")
    stream = _BitStream(blob[4:])
    model = _Model()
    low = 0
    high = _TOP
    value = 0
    for _ in range(32):
        value = (value << 1) | stream.read()

    result = bytearray()
    previous = 0
    older = 0
    for _ in range(length):
        pair = (older << 8) | previous
        row, start, total = model.distribution(pair, previous)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        cumulative = 0
        symbol = 0
        for candidate in range(256):
            frequency = row[start + candidate]
            if target < cumulative + frequency:
                symbol = candidate
                break
            cumulative += frequency
        high = low + (span * (cumulative + frequency) // total) - 1
        low = low + (span * cumulative // total)
        while True:
            if high < _MID:
                pass
            elif low >= _MID:
                low -= _MID
                high -= _MID
                value -= _MID
            elif low >= _LOW_QUARTER and high < _HIGH_QUARTER:
                low -= _LOW_QUARTER
                high -= _LOW_QUARTER
                value -= _LOW_QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | stream.read()
        result.append(symbol)
        model.observe(pair, previous, symbol)
        older, previous = previous, symbol
    return bytes(result)
