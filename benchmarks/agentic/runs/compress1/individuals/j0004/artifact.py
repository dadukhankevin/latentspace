_TOP = 0xFFFFFFFF
_MID = 0x80000000
_LO_QUARTER = 0x40000000
_HI_QUARTER = 0xC0000000
_MODEL_CAP = 49152
_TYPE_INC = 80
_RANK_INC = 24
_RUN_INC = 48
_PREFERRED = (b' etasronildchupm\ngf-.vb,yw0123456789:()[]{}_*#=+/>\\\'"%&!?@$<>|`~'
              b'ABCDEFHIJKLMNOPQRSTUVWXYZ')
_INITIAL_BOOK = []
for _symbol in _PREFERRED + bytes(range(256)):
    if _symbol not in _INITIAL_BOOK:
        _INITIAL_BOOK.append(_symbol)
_INITIAL_BOOK = tuple(_INITIAL_BOOK)


class _Bits:
    def __init__(self, raw=b""):
        self.data = bytearray(raw)
        self.acc = 0
        self.used = 0
        self.pos = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.used += 1
        if self.used == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.used = 0

    def pad(self):
        while self.used:
            self.write(0)

    def read(self):
        if self.pos >= len(self.data) * 8:
            return 0
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class _Arithmetic:
    def __init__(self, raw=b""):
        self.bits = _Bits(raw)
        self.low = 0
        self.high = _TOP
        self.pending = 0
        self.value = 0
        if raw:
            for _ in range(32):
                self.value = (self.value << 1) | self.bits.read()

    def _emit(self, bit):
        self.bits.write(bit)
        while self.pending:
            self.bits.write(1 - bit)
            self.pending -= 1

    def encode(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + span * (cumulative + frequency) // total - 1
        self.low = self.low + span * cumulative // total
        while True:
            if self.high < _MID:
                self._emit(0)
            elif self.low >= _MID:
                self._emit(1)
                self.low -= _MID
                self.high -= _MID
            elif self.low >= _LO_QUARTER and self.high < _HI_QUARTER:
                self.pending += 1
                self.low -= _LO_QUARTER
                self.high -= _LO_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < _LO_QUARTER else 1)
        self.bits.pad()
        return bytes(self.bits.data)

    def decode(self, frequencies, total):
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // span
        cumulative = 0
        for index, frequency in enumerate(frequencies):
            if target < cumulative + frequency:
                break
            cumulative += frequency
        self.high = self.low + span * (cumulative + frequency) // total - 1
        self.low = self.low + span * cumulative // total
        while True:
            if self.high < _MID:
                pass
            elif self.low >= _MID:
                self.low -= _MID
                self.high -= _MID
                self.value -= _MID
            elif self.low >= _LO_QUARTER and self.high < _HI_QUARTER:
                self.low -= _LO_QUARTER
                self.high -= _LO_QUARTER
                self.value -= _LO_QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.bits.read()
        return index


def _new_model():
    return [1, 1], [1] * 255, [1] * 255, 2, 255, 255


def _halve(freq):
    for i, value in enumerate(freq):
        freq[i] = (value + 1) // 2
    return sum(freq)


def _learn(freq, total, index, amount):
    freq[index] += amount
    total += amount
    if total > _MODEL_CAP:
        total = _halve(freq)
    return total


def _rotations(data):
    """Return cyclic rotations in lexicographic order by prefix doubling."""
    size = len(data)
    order = list(range(size))
    rank = list(data)
    width = 1
    while width < size:
        order.sort(key=lambda start: (rank[start], rank[(start + width) % size]))
        next_rank = [0] * size
        classes = 0
        previous = (rank[order[0]], rank[(order[0] + width) % size])
        for start in order:
            pair = (rank[start], rank[(start + width) % size])
            if pair != previous:
                classes += 1
                previous = pair
            next_rank[start] = classes
        rank = next_rank
        if classes == size - 1:
            break
        width <<= 1
    return order


def _forward_transform(data):
    order = _rotations(data)
    last = bytearray(len(data))
    primary = 0
    for row, start in enumerate(order):
        if start == 0:
            primary = row
            last[row] = data[-1]
        else:
            last[row] = data[start - 1]
    return last, primary


def _inverse_transform(last, primary):
    counts = [0] * 256
    occurrence = [0] * len(last)
    for row, symbol in enumerate(last):
        occurrence[row] = counts[symbol]
        counts[symbol] += 1
    first = [0] * 256
    total = 0
    for symbol in range(256):
        first[symbol] = total
        total += counts[symbol]
    restored = bytearray(len(last))
    row = primary
    for at in range(len(last) - 1, -1, -1):
        symbol = last[row]
        restored[at] = symbol
        row = first[symbol] + occurrence[row]
    return bytes(restored)


def _mtf(values):
    book = list(_INITIAL_BOOK)
    ranks = []
    for symbol in values:
        rank = book.index(symbol)
        ranks.append(rank)
        if rank:
            book.insert(0, book.pop(rank))
    return ranks


def _unmtf(ranks):
    book = list(_INITIAL_BOOK)
    values = bytearray()
    for rank in ranks:
        symbol = book[rank]
        values.append(symbol)
        if rank:
            book.insert(0, book.pop(rank))
    return bytes(values)


def _tokens(ranks):
    """Yield (kind, value): kind 0 is a rank, kind 1 is a zero-run length."""
    at = 0
    while at < len(ranks):
        rank = ranks[at]
        if rank:
            yield 0, rank
            at += 1
            continue
        end = at + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        remaining = end - at
        while remaining > 255:
            yield 1, 255
            remaining -= 255
        if remaining:
            yield 1, remaining
        at = end


def _encode_token(coder, kind, value, models):
    type_freq, rank_freq, run_freq, type_total, rank_total, run_total = models
    coder.encode(sum(type_freq[:kind]), type_freq[kind], type_total)
    type_total = _learn(type_freq, type_total, kind, _TYPE_INC)
    if kind == 0:
        index = value - 1
        coder.encode(sum(rank_freq[:index]), rank_freq[index], rank_total)
        rank_total = _learn(rank_freq, rank_total, index, _RANK_INC)
    else:
        index = value - 1
        coder.encode(sum(run_freq[:index]), run_freq[index], run_total)
        run_total = _learn(run_freq, run_total, index, _RUN_INC)
    models[3] = type_total
    models[4] = rank_total
    models[5] = run_total


def compress(data: bytes) -> bytes:
    n = len(data)
    if not n:
        return b"\x00" * 8
    last, primary = _forward_transform(data)
    coder = _Arithmetic()
    models = list(_new_model())
    for kind, value in _tokens(_mtf(last)):
        _encode_token(coder, kind, value, models)
    coded = coder.finish()
    return n.to_bytes(4, "big") + primary.to_bytes(4, "big") + coded


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    if not n:
        return b""
    primary = int.from_bytes(blob[4:8], "big")
    coder = _Arithmetic(blob[8:])
    models = list(_new_model())
    ranks = []
    while len(ranks) < n:
        type_freq, rank_freq, run_freq, type_total, rank_total, run_total = models
        kind = coder.decode(type_freq, type_total)
        type_total = _learn(type_freq, type_total, kind, _TYPE_INC)
        if kind == 0:
            index = coder.decode(rank_freq, rank_total)
            rank_total = _learn(rank_freq, rank_total, index, _RANK_INC)
            ranks.append(index + 1)
        else:
            index = coder.decode(run_freq, run_total)
            run_total = _learn(run_freq, run_total, index, _RUN_INC)
            ranks.extend([0] * (index + 1))
        models[3] = type_total
        models[4] = rank_total
        models[5] = run_total
    last = _unmtf(ranks[:n])
    return _inverse_transform(last, primary)
