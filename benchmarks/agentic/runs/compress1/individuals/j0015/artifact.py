"""BWT/MTF rank-run coding with sparse second-order token prediction."""

_TOP = 0xFFFFFFFF
_HALF = 0x80000000
_QUARTER = 0x40000000
_THREE_QUARTER = 0xC0000000
_TOKEN_COUNT = 512
_BACKOFF = 32
_ONE_STEP = 16
_ONE_LIMIT = 16384
_TWO_LIMIT = 1024


class _BitStream:
    def __init__(self, raw=b""):
        self.buf = bytearray(raw)
        self.acc = 0
        self.have = 0
        self.index = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.have += 1
        if self.have == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.have = 0

    def finish(self):
        while self.have:
            self.put(0)

    def get(self):
        if self.index >= len(self.buf) * 8:
            return 0
        byte = self.buf[self.index >> 3]
        bit = (byte >> (7 - (self.index & 7))) & 1
        self.index += 1
        return bit


def _rotation_order(data):
    """Return cyclic rotations in lexicographic order."""
    size = len(data)
    order = list(range(size))
    ranks = list(data)
    width = 1
    while width < size:
        order.sort(key=lambda pos: (ranks[pos], ranks[(pos + width) % size]))
        next_ranks = [0] * size
        group = 0
        previous = None
        for pos in order:
            pair = (ranks[pos], ranks[(pos + width) % size])
            if previous is not None and pair != previous:
                group += 1
            next_ranks[pos] = group
            previous = pair
        ranks = next_ranks
        if group == size - 1:
            break
        width <<= 1
    return order


def _forward_bwt(data):
    size = len(data)
    order = _rotation_order(data)
    last = bytearray(size)
    primary = 0
    for row, start in enumerate(order):
        if start == 0:
            primary = row
            last[row] = data[-1]
        else:
            last[row] = data[start - 1]
    return bytes(last), primary


def _inverse_bwt(last, primary):
    counts = [0] * 256
    occurrence = [0] * len(last)
    for index, symbol in enumerate(last):
        occurrence[index] = counts[symbol]
        counts[symbol] += 1
    first = [0] * 256
    offset = 0
    for symbol in range(256):
        first[symbol] = offset
        offset += counts[symbol]
    restored = bytearray(len(last))
    row = primary
    for index in range(len(last) - 1, -1, -1):
        symbol = last[row]
        restored[index] = symbol
        row = first[symbol] + occurrence[row]
    return bytes(restored)


def _forward_mtf(data):
    table = list(range(256))
    ranks = []
    for symbol in data:
        rank = table.index(symbol)
        ranks.append(rank)
        if rank:
            table.pop(rank)
            table.insert(0, symbol)
    return ranks


def _inverse_mtf(ranks):
    table = list(range(256))
    data = bytearray()
    for rank in ranks:
        symbol = table[rank]
        data.append(symbol)
        if rank:
            table.pop(rank)
            table.insert(0, symbol)
    return bytes(data)


def _make_tokens(ranks):
    tokens = []
    at = 0
    while at < len(ranks):
        rank = ranks[at]
        if rank != 0:
            tokens.append(rank)
            at += 1
            continue
        end = at + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        run = end - at
        while run > 255:
            tokens.append(511)
            run -= 255
        if run:
            tokens.append(256 + run)
        at = end
    return tokens


def _initial_row():
    row = [1] * _TOKEN_COUNT
    row[0] = 0
    row[256] = 0
    return row


class _TokenModel:
    def __init__(self):
        self.one = [_initial_row() for _ in range(_TOKEN_COUNT)]
        self.one_totals = [510] * _TOKEN_COUNT
        self.two = {}

    def _pair(self, old, recent):
        key = (old, recent)
        pair = self.two.get(key)
        if pair is None:
            pair = [[0] * _TOKEN_COUNT, 0]
            self.two[key] = pair
        return pair

    def span_for(self, old, recent, symbol):
        row = self.one[recent]
        one_total = self.one_totals[recent]
        direct = self._pair(old, recent)
        direct_row = direct[0]
        direct_total = direct[1]
        one_cumulative = sum(row[:symbol])
        direct_cumulative = sum(direct_row[:symbol])
        frequency = _BACKOFF * row[symbol] + one_total * direct_row[symbol]
        total = _BACKOFF * one_total + one_total * direct_total
        return (_BACKOFF * one_cumulative + one_total * direct_cumulative,
                frequency, total)

    def symbol_for(self, old, recent, target):
        row = self.one[recent]
        one_total = self.one_totals[recent]
        direct = self._pair(old, recent)
        direct_row = direct[0]
        direct_total = direct[1]
        cumulative = 0
        for symbol in range(_TOKEN_COUNT):
            frequency = _BACKOFF * row[symbol] + one_total * direct_row[symbol]
            if frequency and target < cumulative + frequency:
                return symbol, cumulative, frequency, (
                    _BACKOFF * one_total + one_total * direct_total)
            cumulative += frequency
        raise ValueError("arithmetic target outside token model")

    def observe(self, old, recent, symbol):
        row = self.one[recent]
        row[symbol] += _ONE_STEP
        total = self.one_totals[recent] + _ONE_STEP
        if total > _ONE_LIMIT:
            total = 0
            for index, value in enumerate(row):
                value = (value + 1) >> 1
                row[index] = value
                total += value
        self.one_totals[recent] = total

        direct = self._pair(old, recent)
        direct[0][symbol] += 1
        direct[1] += 1
        if direct[1] > _TWO_LIMIT:
            fresh_total = 0
            for index, value in enumerate(direct[0]):
                value = (value + 1) >> 1
                direct[0][index] = value
                fresh_total += value
            direct[1] = fresh_total


def _write_arithmetic(tokens):
    bits = _BitStream()
    model = _TokenModel()
    low = 0
    high = _TOP
    pending = 0
    old = 0
    recent = 0

    def emit(bit):
        nonlocal pending
        bits.put(bit)
        while pending:
            bits.put(bit ^ 1)
            pending -= 1

    for symbol in tokens:
        cumulative, frequency, total = model.span_for(old, recent, symbol)
        width = high - low + 1
        high = low + (width * (cumulative + frequency) // total) - 1
        low = low + (width * cumulative // total)
        while True:
            if high < _HALF:
                emit(0)
            elif low >= _HALF:
                emit(1)
                low -= _HALF
                high -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                pending += 1
                low -= _QUARTER
                high -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        model.observe(old, recent, symbol)
        old, recent = recent, symbol

    pending += 1
    emit(0 if low < _QUARTER else 1)
    bits.finish()
    return bytes(bits.buf)


def _read_arithmetic(blob, expanded_length):
    bits = _BitStream(blob)
    model = _TokenModel()
    low = 0
    high = _TOP
    value = 0
    for _ in range(32):
        value = (value << 1) | bits.get()
    ranks = []
    old = 0
    recent = 0
    while len(ranks) < expanded_length:
        width = high - low + 1
        total = model.span_for(old, recent, 1)[2]
        target = ((value - low + 1) * total - 1) // width
        symbol, cumulative, frequency, total = model.symbol_for(
            old, recent, target)
        high = low + (width * (cumulative + frequency) // total) - 1
        low = low + (width * cumulative // total)
        while True:
            if high < _HALF:
                pass
            elif low >= _HALF:
                low -= _HALF
                high -= _HALF
                value -= _HALF
            elif low >= _QUARTER and high < _THREE_QUARTER:
                low -= _QUARTER
                high -= _QUARTER
                value -= _QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | bits.get()
        if symbol >= 257:
            ranks.extend([0] * (symbol - 256))
        elif symbol != 0:
            ranks.append(symbol)
        else:
            raise ValueError("decoded impossible token")
        model.observe(old, recent, symbol)
        old, recent = recent, symbol
    return ranks[:expanded_length]


def compress(data: bytes) -> bytes:
    if not data:
        return b"\x00" * 8
    last, primary = _forward_bwt(data)
    ranks = _forward_mtf(last)
    tokens = _make_tokens(ranks)
    coded = _write_arithmetic(tokens)
    return (len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + coded)


def decompress(blob: bytes) -> bytes:
    size = int.from_bytes(blob[:4], "big")
    if size == 0:
        return b""
    primary = int.from_bytes(blob[4:8], "big")
    ranks = _read_arithmetic(blob[8:], size)
    last = _inverse_mtf(ranks)
    return _inverse_bwt(last, primary)
