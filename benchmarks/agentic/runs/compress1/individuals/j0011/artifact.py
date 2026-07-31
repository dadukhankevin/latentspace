"""A self-contained BWT/MTF compressor with synchronized adaptive models."""

FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1


class _Bits:
    def __init__(self, data=b""):
        self.data = bytearray(data)
        self.acc = 0
        self.nbits = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.data.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def finish(self):
        while self.nbits:
            self.put(0)

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        b = self.data[self.pos >> 3]
        bit = (b >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class _Encoder:
    def __init__(self):
        self.bits = _Bits()
        self.low = 0
        self.high = FULL
        self.pending = 0

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(bit ^ 1)
            self.pending -= 1

    def symbol(self, cum, freq, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cum + freq)) // total - 1
        self.low = self.low + (span * cum) // total
        while True:
            if self.high < HALF:
                self._emit(0)
            elif self.low >= HALF:
                self._emit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= Q1 and self.high < Q3:
                self.pending += 1
                self.low -= Q1
                self.high -= Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < Q1 else 1)
        self.bits.finish()
        return bytes(self.bits.data)


class _Decoder:
    def __init__(self, data):
        self.bits = _Bits(data)
        self.low = 0
        self.high = FULL
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.get()

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def symbol(self, cum, freq, total):
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
            elif self.low >= Q1 and self.high < Q3:
                self.low -= Q1
                self.high -= Q1
                self.value -= Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.bits.get()


class _FlatModel:
    def __init__(self, size, increment):
        self.counts = [1] * size
        self.total = size
        self.increment = increment

    def span(self, symbol):
        cum = 0
        for i in range(symbol):
            cum += self.counts[i]
        return cum, self.counts[symbol], self.total

    def find(self, target):
        cum = 0
        for i, freq in enumerate(self.counts):
            if target < cum + freq:
                return i, cum, freq, self.total
            cum += freq
        i = len(self.counts) - 1
        return i, self.total - self.counts[i], self.counts[i], self.total

    def update(self, symbol):
        self.counts[symbol] += self.increment
        self.total += self.increment


class _TypeModel:
    def __init__(self):
        self.rows = [[1, 1], [1, 1]]
        self.totals = [2, 2]
        self.increment = 4

    def span(self, previous, symbol):
        row = self.rows[previous]
        return (0 if symbol == 0 else row[0], row[symbol], self.totals[previous])

    def find(self, previous, target):
        row = self.rows[previous]
        if target < row[0]:
            return 0, 0, row[0], self.totals[previous]
        return 1, row[0], row[1], self.totals[previous]

    def update(self, previous, symbol):
        self.rows[previous][symbol] += self.increment
        self.totals[previous] += self.increment


class _RankModel:
    """Exact previous-rank contexts with an adaptive unigram backoff."""
    def __init__(self):
        self.global_counts = [1] * 255
        self.global_total = 255
        self.rows = [None] * 256
        self.row_totals = [0] * 256
        self.row_weight = 8
        self.increment = 4

    def span(self, previous, symbol):
        row = self.rows[previous]
        if row is None:
            cum = 0
            for i in range(symbol):
                cum += self.global_counts[i]
            return cum, self.global_counts[symbol], self.global_total
        cum = 0
        for i in range(symbol):
            cum += self.row_weight * row[i] + self.global_counts[i]
        freq = self.row_weight * row[symbol] + self.global_counts[symbol]
        return cum, freq, self.row_weight * self.row_totals[previous] + self.global_total

    def find(self, previous, target):
        row = self.rows[previous]
        if row is None:
            cum = 0
            for i, freq in enumerate(self.global_counts):
                if target < cum + freq:
                    return i, cum, freq, self.global_total
                cum += freq
            i = 254
            return i, self.global_total - self.global_counts[i], self.global_counts[i], self.global_total
        total = self.row_weight * self.row_totals[previous] + self.global_total
        cum = 0
        for i in range(255):
            freq = self.row_weight * row[i] + self.global_counts[i]
            if target < cum + freq:
                return i, cum, freq, total
            cum += freq
        i = 254
        freq = self.row_weight * row[i] + self.global_counts[i]
        return i, total - freq, freq, total

    def update(self, previous, symbol):
        row = self.rows[previous]
        if row is None:
            row = [0] * 255
            self.rows[previous] = row
        row[symbol] += self.increment
        self.row_totals[previous] += self.increment
        self.global_counts[symbol] += self.increment
        self.global_total += self.increment


def _suffix_array(data):
    n = len(data)
    classes = list(data)
    order = list(range(n))
    step = 1
    while True:
        order.sort(key=lambda i: (classes[i], classes[(i + step) % n]))
        new_classes = [0] * n
        for j in range(1, n):
            left = order[j - 1]
            right = order[j]
            new_classes[right] = new_classes[left] + (
                (classes[left], classes[(left + step) % n])
                != (classes[right], classes[(right + step) % n])
            )
        classes = new_classes
        if classes[order[-1]] == n - 1 or step >= n:
            return order
        step <<= 1


def _forward(data):
    n = len(data)
    order = _suffix_array(data)
    primary = order.index(0)
    bwt = bytes(data[(i - 1) % n] for i in order)
    mtf = list(range(256))
    ranks = []
    for value in bwt:
        rank = mtf.index(value)
        ranks.append(rank)
        mtf.pop(rank)
        mtf.insert(0, value)
    events = []
    i = 0
    while i < n:
        if ranks[i]:
            events.append((0, ranks[i]))
            i += 1
        else:
            j = i + 1
            while j < n and ranks[j] == 0:
                j += 1
            run = j - i
            while run > 255:
                events.append((1, 255))
                run -= 255
            events.append((1, run))
            i = j
    return primary, events


def _inverse_bwt(bwt, primary):
    n = len(bwt)
    counts = [0] * 256
    occurrence = [0] * n
    for i, value in enumerate(bwt):
        occurrence[i] = counts[value]
        counts[value] += 1
    starts = [0] * 256
    total = 0
    for value in range(256):
        starts[value] = total
        total += counts[value]
    row = primary
    result = bytearray(n)
    for i in range(n - 1, -1, -1):
        value = bwt[row]
        result[i] = value
        row = starts[value] + occurrence[row]
    return bytes(result)


def _decode_events(decoder, n):
    type_model = _TypeModel()
    rank_model = _RankModel()
    zero_model = _FlatModel(255, 24)
    previous_type = 0
    previous_rank = 1
    ranks = []
    while len(ranks) < n:
        target = decoder.target(type_model.totals[previous_type])
        kind, cum, freq, total = type_model.find(previous_type, target)
        decoder.symbol(cum, freq, total)
        type_model.update(previous_type, kind)
        if kind == 0:
            target = decoder.target(rank_model.span(previous_rank, 0)[2])
            rank, cum, freq, total = rank_model.find(previous_rank, target)
            decoder.symbol(cum, freq, total)
            rank_model.update(previous_rank, rank)
            rank += 1
            ranks.append(rank)
            previous_rank = rank
        else:
            total = zero_model.total
            target = decoder.target(total)
            length, cum, freq, total = zero_model.find(target)
            decoder.symbol(cum, freq, total)
            zero_model.update(length)
            ranks.extend([0] * (length + 1))
        previous_type = kind
    return ranks[:n]


def compress(data: bytes) -> bytes:
    if not data:
        return (0).to_bytes(4, "big") + (0).to_bytes(4, "big")
    primary, events = _forward(data)
    encoder = _Encoder()
    type_model = _TypeModel()
    rank_model = _RankModel()
    zero_model = _FlatModel(255, 24)
    previous_type = 0
    previous_rank = 1
    for kind, value in events:
        cum, freq, total = type_model.span(previous_type, kind)
        encoder.symbol(cum, freq, total)
        type_model.update(previous_type, kind)
        if kind == 0:
            symbol = value - 1
            cum, freq, total = rank_model.span(previous_rank, symbol)
            encoder.symbol(cum, freq, total)
            rank_model.update(previous_rank, symbol)
            previous_rank = value
        else:
            symbol = value - 1
            cum, freq, total = zero_model.span(symbol)
            encoder.symbol(cum, freq, total)
            zero_model.update(symbol)
        previous_type = kind
    payload = encoder.finish()
    return len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + payload


def decompress(blob: bytes) -> bytes:
    if len(blob) < 8:
        return b""
    n = int.from_bytes(blob[:4], "big")
    primary = int.from_bytes(blob[4:8], "big")
    if n == 0:
        return b""
    decoder = _Decoder(blob[8:])
    ranks = _decode_events(decoder, n)
    mtf = list(range(256))
    bwt = bytearray()
    for rank in ranks:
        value = mtf[rank]
        bwt.append(value)
        mtf.pop(rank)
        mtf.insert(0, value)
    return _inverse_bwt(bytes(bwt), primary)
