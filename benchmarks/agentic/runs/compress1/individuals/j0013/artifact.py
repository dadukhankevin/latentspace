_MAX_CODE = 0xFFFFFFFF
_HALF_CODE = 0x80000000
_QTR_CODE = 0x40000000
_THREE_QTR_CODE = 0xC0000000
_TOKEN_COUNT = 512
_UPDATE = 19
_RESCALE_AT = 65536


class _BitWriter:
    def __init__(self):
        self.buffer = bytearray()
        self.value = 0
        self.pending_bits = 0

    def add(self, bit):
        self.value = (self.value << 1) | bit
        self.pending_bits += 1
        if self.pending_bits == 8:
            self.buffer.append(self.value)
            self.value = 0
            self.pending_bits = 0

    def close(self):
        while self.pending_bits:
            self.add(0)
        return bytes(self.buffer)


class _BitReader:
    def __init__(self, payload):
        self.payload = payload
        self.offset = 0

    def take(self):
        if self.offset >= len(self.payload) * 8:
            return 0
        byte = self.payload[self.offset >> 3]
        bit = (byte >> (7 - (self.offset & 7))) & 1
        self.offset += 1
        return bit


def _cyclic_order(data):
    count = len(data)
    order = list(range(count))
    ranks = list(data)
    width = 1
    while width < count:
        order.sort(key=lambda start: (ranks[start], ranks[(start + width) % count]))
        fresh = [0] * count
        cls = 0
        previous = (ranks[order[0]], ranks[(order[0] + width) % count])
        fresh[order[0]] = 0
        for start in order[1:]:
            current = (ranks[start], ranks[(start + width) % count])
            if current != previous:
                cls += 1
                previous = current
            fresh[start] = cls
        ranks = fresh
        if cls == count - 1:
            break
        width <<= 1
    return order


def _forward_transform(data):
    order = _cyclic_order(data)
    last_column = bytearray(len(data))
    primary = 0
    for row, start in enumerate(order):
        if start == 0:
            primary = row
            last_column[row] = data[-1]
        else:
            last_column[row] = data[start - 1]
    return bytes(last_column), primary


def _inverse_transform(last_column, primary):
    frequencies = [0] * 256
    occurrence = [0] * len(last_column)
    for row, symbol in enumerate(last_column):
        occurrence[row] = frequencies[symbol]
        frequencies[symbol] += 1
    starts = [0] * 256
    cursor = 0
    for symbol in range(256):
        starts[symbol] = cursor
        cursor += frequencies[symbol]
    restored = bytearray(len(last_column))
    row = primary
    for pos in range(len(last_column) - 1, -1, -1):
        symbol = last_column[row]
        restored[pos] = symbol
        row = starts[symbol] + occurrence[row]
    return bytes(restored)


def _front_code(data):
    book = list(range(256))
    positions = list(range(256))
    result = []
    for symbol in data:
        rank = positions[symbol]
        result.append(rank)
        if rank:
            old = book[:rank]
            book[1:rank + 1] = old
            book[0] = symbol
            for pos in range(1, rank + 1):
                positions[book[pos]] = pos
            positions[symbol] = 0
    return result


def _front_decode(ranks):
    book = list(range(256))
    result = bytearray()
    for rank in ranks:
        symbol = book[rank]
        result.append(symbol)
        if rank:
            old = book[:rank]
            book[1:rank + 1] = old
            book[0] = symbol
    return bytes(result)


def _make_tokens(ranks):
    tokens = []
    pos = 0
    while pos < len(ranks):
        rank = ranks[pos]
        if rank:
            tokens.append(rank)
            pos += 1
            continue
        end = pos + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        run = end - pos
        while run > 255:
            tokens.append(511)
            run -= 255
        if run:
            tokens.append(256 + run)
        pos = end
    return tokens


def _token_model():
    counts = [0] * _TOKEN_COUNT
    for rank in range(1, 256):
        counts[rank] = max(1, round(384 / (rank + 1)))
    for run in range(1, 256):
        counts[256 + run] = max(1, round(38.4 / (run + 1)))
    return counts


def _encode_tokens(tokens):
    counts = _token_model()
    writer = _BitWriter()
    low = 0
    high = _MAX_CODE
    underflow = 0

    def emit(bit):
        nonlocal underflow
        writer.add(bit)
        while underflow:
            writer.add(1 - bit)
            underflow -= 1

    for token in tokens:
        total = sum(counts)
        cumulative = sum(counts[:token])
        span = high - low + 1
        high = low + (span * (cumulative + counts[token]) // total) - 1
        low = low + (span * cumulative // total)
        while True:
            if high < _HALF_CODE:
                emit(0)
            elif low >= _HALF_CODE:
                emit(1)
                low -= _HALF_CODE
                high -= _HALF_CODE
            elif low >= _QTR_CODE and high < _THREE_QTR_CODE:
                underflow += 1
                low -= _QTR_CODE
                high -= _QTR_CODE
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        counts[token] += _UPDATE
        if total + _UPDATE > _RESCALE_AT:
            counts = [(value + 1) // 2 for value in counts]

    underflow += 1
    emit(0 if low < _QTR_CODE else 1)
    return writer.close()


def _decode_tokens(payload, rank_count):
    counts = _token_model()
    reader = _BitReader(payload)
    value = 0
    for _ in range(32):
        value = (value << 1) | reader.take()
    low = 0
    high = _MAX_CODE
    ranks = []

    while len(ranks) < rank_count:
        total = sum(counts)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        cumulative = 0
        token = 1
        for token in range(_TOKEN_COUNT):
            frequency = counts[token]
            if target < cumulative + frequency:
                break
            cumulative += frequency
        frequency = counts[token]
        high = low + (span * (cumulative + frequency) // total) - 1
        low = low + (span * cumulative // total)
        while True:
            if high < _HALF_CODE:
                pass
            elif low >= _HALF_CODE:
                low -= _HALF_CODE
                high -= _HALF_CODE
                value -= _HALF_CODE
            elif low >= _QTR_CODE and high < _THREE_QTR_CODE:
                low -= _QTR_CODE
                high -= _QTR_CODE
                value -= _QTR_CODE
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            value = (value << 1) | reader.take()
        counts[token] += _UPDATE
        if total + _UPDATE > _RESCALE_AT:
            counts = [(item + 1) // 2 for item in counts]
        if token >= 256:
            ranks.extend([0] * (token - 256))
        else:
            ranks.append(token)
    return ranks[:rank_count]


def compress(data: bytes) -> bytes:
    if not data:
        return b"\x00" * 8
    last_column, primary = _forward_transform(data)
    ranks = _front_code(last_column)
    tokens = _make_tokens(ranks)
    encoded = _encode_tokens(tokens)
    return (len(data).to_bytes(4, "big") + primary.to_bytes(4, "big") + encoded)


def decompress(blob: bytes) -> bytes:
    length = int.from_bytes(blob[:4], "big")
    if length == 0:
        return b""
    primary = int.from_bytes(blob[4:8], "big")
    ranks = _decode_tokens(blob[8:], length)
    last_column = _front_decode(ranks)
    return _inverse_transform(last_column, primary)
