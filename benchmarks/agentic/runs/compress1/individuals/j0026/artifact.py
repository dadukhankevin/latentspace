"""A compact text-oriented compressor built around a reversible BWT."""


_WORD = 0xFFFFFFFF
_TOP_BYTE = 0xFF000000
_ESCAPE_STEP = 12
_MODEL_LIMIT = 18000


class _Emitter:
    def __init__(self):
        self.low = 0
        self.high = _WORD
        self.output = bytearray()

    def symbol(self, before, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (before + width) // total) - 1
        self.low = self.low + (span * before // total)
        while ((self.low ^ self.high) & _TOP_BYTE) == 0:
            self.output.append(self.high >> 24)
            self.low = (self.low << 8) & _WORD
            self.high = ((self.high << 8) | 255) & _WORD

    def close(self):
        for _ in range(4):
            self.output.append(self.low >> 24)
            self.low = (self.low << 8) & _WORD
        return bytes(self.output)


class _Reader:
    def __init__(self, payload):
        self.payload = payload
        self.position = 0
        self.low = 0
        self.high = _WORD
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._byte()

    def _byte(self):
        if self.position >= len(self.payload):
            return 0
        value = self.payload[self.position]
        self.position += 1
        return value

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def symbol(self, before, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (before + width) // total) - 1
        self.low = self.low + (span * before // total)
        while ((self.low ^ self.high) & _TOP_BYTE) == 0:
            self.low = (self.low << 8) & _WORD
            self.high = ((self.high << 8) | 255) & _WORD
            self.code = ((self.code << 8) & _WORD) | self._byte()


def _cumulative(frequencies, value):
    total = 0
    for index in range(value):
        total += frequencies[index]
    return total


def _adapt(frequencies, total, value):
    frequencies[value] += _ESCAPE_STEP
    total += _ESCAPE_STEP
    if total > _MODEL_LIMIT:
        frequencies = [(item + 1) // 2 for item in frequencies]
        total = sum(frequencies)
    return frequencies, total


def _pack_symbols(symbols, terminator):
    """Arithmetic-code symbols, ending with a self-delimiting marker."""
    alphabet_size = terminator + 1
    frequencies = [1] * alphabet_size
    total = alphabet_size
    coder = _Emitter()
    for value in symbols:
        before = _cumulative(frequencies, value)
        coder.symbol(before, frequencies[value], total)
        frequencies, total = _adapt(frequencies, total, value)
    before = _cumulative(frequencies, terminator)
    coder.symbol(before, frequencies[terminator], total)
    return coder.close()


def _unpack_symbols(payload, terminator):
    alphabet_size = terminator + 1
    frequencies = [1] * alphabet_size
    total = alphabet_size
    coder = _Reader(payload)
    result = []
    while True:
        target = coder.target(total)
        running = 0
        value = -1
        for index, frequency in enumerate(frequencies):
            if target < running + frequency:
                value = index
                coder.symbol(running, frequency, total)
                break
            running += frequency
        if value < 0:
            raise ValueError("arithmetic symbol outside model")
        frequencies, total = _adapt(frequencies, total, value)
        if value == terminator:
            return result
        result.append(value)


def _suffix_bwt(data):
    values = list(data) + [256]
    size = len(values)
    order = list(range(size))
    ranks = values[:]
    distance = 1
    while distance < size:
        order.sort(key=lambda start: (ranks[start],
                                      ranks[(start + distance) % size]))
        fresh = [0] * size
        classes = 0
        previous = None
        for start in order:
            pair = (ranks[start], ranks[(start + distance) % size])
            if previous is not None and pair != previous:
                classes += 1
            fresh[start] = classes
            previous = pair
        ranks = fresh
        if classes == size - 1:
            break
        distance <<= 1

    primary = 0
    last_column = [0] * size
    for row, start in enumerate(order):
        if start == 0:
            primary = row
        last_column[row] = values[(start - 1) % size]
    return last_column, primary


def _inverse_bwt(last_column, primary):
    size = len(last_column)
    if not size or primary >= size:
        raise ValueError("invalid transform header")
    counts = [0] * 257
    for value in last_column:
        if value < 0 or value > 256:
            raise ValueError("invalid transform symbol")
        counts[value] += 1
    starts = [0] * 257
    offset = 0
    for value in range(257):
        starts[value] = offset
        offset += counts[value]
    seen = [0] * 257
    links = [0] * size
    for row, value in enumerate(last_column):
        links[row] = starts[value] + seen[value]
        seen[value] += 1

    row = primary
    restored = [0] * size
    for index in range(size - 1, -1, -1):
        restored[index] = last_column[row]
        row = links[row]
    if restored[-1] != 256:
        raise ValueError("transform sentinel missing")
    return bytes(restored[:-1])


def _move_front(last_column, alphabet):
    table = list(alphabet)
    ranks = []
    for value in last_column:
        try:
            rank = table.index(value)
        except ValueError as exc:
            raise ValueError("transform alphabet mismatch") from exc
        ranks.append(rank)
        table.pop(rank)
        table.insert(0, value)
    return ranks


def _undo_move_front(ranks, alphabet):
    table = list(alphabet)
    last_column = []
    for rank in ranks:
        if rank < 0 or rank >= len(table):
            raise ValueError("invalid move-front rank")
        value = table.pop(rank)
        table.insert(0, value)
        last_column.append(value)
    return last_column


def _zero_runs(ranks):
    """Represent each zero run by the low-to-high bits of run_length - 1."""
    stream = []
    index = 0
    while index < len(ranks):
        rank = ranks[index]
        if rank:
            stream.append(rank + 1)
            index += 1
            continue
        end = index + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        number = end - index - 1
        while True:
            stream.append(number & 1)
            number >>= 1
            if number == 0:
                break
        index = end
    return stream


def _restore_zero_runs(stream):
    ranks = []
    index = 0
    while index < len(stream):
        value = stream[index]
        index += 1
        if value >= 2:
            ranks.append(value - 1)
            continue
        number = 0
        shift = 0
        while True:
            number |= (value & 1) << shift
            shift += 1
            if index >= len(stream) or stream[index] >= 2:
                break
            value = stream[index]
            index += 1
        ranks.extend([0] * (number + 1))
    return ranks


def _presence_map(values):
    bitmap = bytearray(33)
    for value in values:
        bitmap[value >> 3] |= 1 << (value & 7)
    return bytes(bitmap)


def _alphabet_from_map(bitmap):
    alphabet = []
    for value in range(257):
        if bitmap[value >> 3] & (1 << (value & 7)):
            alphabet.append(value)
    if not alphabet or alphabet[-1] != 256:
        raise ValueError("transform alphabet lacks sentinel")
    return alphabet


def _bwt_blob(data, sparse):
    last_column, primary = _suffix_bwt(data)
    if sparse:
        alphabet = sorted(set(last_column))
        prefix = b"S" + primary.to_bytes(4, "big")
        prefix += _presence_map(alphabet)
    else:
        alphabet = list(range(257))
        prefix = b"F" + primary.to_bytes(4, "big")
    ranks = _move_front(last_column, alphabet)
    stream = _zero_runs(ranks)
    # rank+1 can equal len(alphabet); the next value is the terminator.
    terminator = len(alphabet) + 1
    return prefix + _pack_symbols(stream, terminator)


def _unbwt_blob(blob):
    tag = blob[:1]
    if tag == b"F":
        if len(blob) < 5:
            raise ValueError("truncated full transform")
        primary = int.from_bytes(blob[1:5], "big")
        alphabet = list(range(257))
        payload = blob[5:]
    elif tag == b"S":
        if len(blob) < 38:
            raise ValueError("truncated sparse transform")
        primary = int.from_bytes(blob[1:5], "big")
        alphabet = _alphabet_from_map(blob[5:38])
        payload = blob[38:]
    else:
        raise ValueError("unknown transform tag")
    terminator = len(alphabet) + 1
    stream = _unpack_symbols(payload, terminator)
    ranks = _restore_zero_runs(stream)
    last_column = _undo_move_front(ranks, alphabet)
    return _inverse_bwt(last_column, primary)


def compress(data: bytes) -> bytes:
    raw = bytes(data)
    raw_blob = b"R" + len(raw).to_bytes(4, "big") + raw
    full_blob = _bwt_blob(raw, False)
    sparse_blob = _bwt_blob(raw, True)
    return min((raw_blob, full_blob, sparse_blob), key=len)


def decompress(blob: bytes) -> bytes:
    if not blob:
        raise ValueError("empty compressed stream")
    if blob[0] == 82:
        if len(blob) < 5:
            raise ValueError("truncated raw stream")
        size = int.from_bytes(blob[1:5], "big")
        result = blob[5:]
        if len(result) != size:
            raise ValueError("raw size mismatch")
        return bytes(result)
    if blob[0] in (70, 83):
        return _unbwt_blob(blob)
    raise ValueError("unknown compressed stream")
