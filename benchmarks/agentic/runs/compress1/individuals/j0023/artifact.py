"""Run-aware BWT compressor with two adaptive range-coded streams."""


_WORD = 0xFFFFFFFF
_TOP_BYTE = 0xFF000000
_PREFERRED = (32, 10, 13, 46, 44, 59, 58, 33, 63, 45, 40, 41,
              91, 93, 123, 125, 95, 39, 34, 47, 92) + tuple(range(48, 58))
_PREFERRED += tuple(b"etaoinshrdlucmfwypvbgkjqxz")


class _Encoder:
    def __init__(self):
        self.low = 0
        self.high = _WORD
        self.out = bytearray()

    def put(self, start, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (start + width) // total) - 1
        self.low = self.low + (span * start // total)
        while ((self.low ^ self.high) & _TOP_BYTE) == 0:
            self.out.append(self.high >> 24)
            self.low = (self.low << 8) & _WORD
            self.high = ((self.high << 8) | 255) & _WORD

    def finish(self):
        for _ in range(4):
            self.out.append(self.low >> 24)
            self.low = (self.low << 8) & _WORD
        return bytes(self.out)


class _Decoder:
    def __init__(self, blob):
        self.blob = blob
        self.pos = 0
        self.low = 0
        self.high = _WORD
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._read()

    def _read(self):
        if self.pos >= len(self.blob):
            return 0
        value = self.blob[self.pos]
        self.pos += 1
        return value

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def take(self, start, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (start + width) // total) - 1
        self.low = self.low + (span * start // total)
        while ((self.low ^ self.high) & _TOP_BYTE) == 0:
            self.low = (self.low << 8) & _WORD
            self.high = ((self.high << 8) | 255) & _WORD
            self.code = ((self.code << 8) & _WORD) | self._read()


def _suffix_bwt(data):
    values = list(data) + [256]
    size = len(values)
    order = list(range(size))
    rank = values[:]
    step = 1
    while step < size:
        order.sort(key=lambda at: (rank[at], rank[(at + step) % size]))
        new_rank = [0] * size
        classes = 0
        prior = None
        for at in order:
            pair = (rank[at], rank[(at + step) % size])
            if prior is not None and pair != prior:
                classes += 1
            new_rank[at] = classes
            prior = pair
        rank = new_rank
        if classes == size - 1:
            break
        step <<= 1

    primary = 0
    last = [0] * size
    for row, at in enumerate(order):
        if at == 0:
            primary = row
        last[row] = values[(at - 1) % size]
    return last, primary


def _split_runs(last):
    runs = []
    at = 0
    while at < len(last):
        end = at + 1
        symbol = last[at]
        while end < len(last) and last[end] == symbol:
            end += 1
        remaining = end - at
        while remaining > 255:
            runs.append((symbol, 255))
            remaining -= 255
        runs.append((symbol, remaining))
        at = end
    return runs


def _mtf_runs(runs):
    used = [False] * 257
    alphabet = []
    for symbol in _PREFERRED + tuple(range(257)):
        if not used[symbol]:
            used[symbol] = True
            alphabet.append(symbol)
    ranks = []
    lengths = []
    for symbol, length in runs:
        rank = alphabet.index(symbol)
        ranks.append(rank)
        lengths.append(length)
        alphabet.pop(rank)
        alphabet.insert(0, symbol)
    return ranks, lengths


def _encode_symbols(symbols, alphabet_size, increment):
    encoder = _Encoder()
    frequencies = [1] * alphabet_size
    total = alphabet_size
    for symbol in symbols:
        cumulative = 0
        for value in range(symbol):
            cumulative += frequencies[value]
        encoder.put(cumulative, frequencies[symbol], total)
        frequencies[symbol] += increment
        total += increment
        if total > 32768:
            total = 0
            for value in range(alphabet_size):
                frequencies[value] = (frequencies[value] + 1) // 2
                total += frequencies[value]
    return encoder.finish()


def _decode_symbols(blob, count, alphabet_size, increment):
    decoder = _Decoder(blob)
    frequencies = [1] * alphabet_size
    total = alphabet_size
    result = []
    for _ in range(count):
        target = decoder.target(total)
        cumulative = 0
        symbol = -1
        for value, frequency in enumerate(frequencies):
            if target < cumulative + frequency:
                symbol = value
                decoder.take(cumulative, frequency, total)
                break
            cumulative += frequency
        if symbol < 0:
            raise ValueError("range stream escaped its alphabet")
        result.append(symbol)
        frequencies[symbol] += increment
        total += increment
        if total > 32768:
            total = 0
            for value in range(alphabet_size):
                frequencies[value] = (frequencies[value] + 1) // 2
                total += frequencies[value]
    return result


def _encode_bwt(data):
    last, primary = _suffix_bwt(data)
    runs = _split_runs(last)
    ranks, lengths = _mtf_runs(runs)
    max_length = max(lengths)
    encoder = _Encoder()
    rank_frequencies = [1] * 257
    rank_total = 257
    length_frequencies = [1] * max_length
    length_total = max_length
    for rank, length in zip(ranks, lengths):
        cumulative = 0
        for value in range(rank):
            cumulative += rank_frequencies[value]
        encoder.put(cumulative, rank_frequencies[rank], rank_total)
        rank_frequencies[rank] += 12
        rank_total += 12
        if rank_total > 32768:
            rank_total = 0
            for value in range(257):
                rank_frequencies[value] = (rank_frequencies[value] + 1) // 2
                rank_total += rank_frequencies[value]

        symbol = length - 1
        cumulative = 0
        for value in range(symbol):
            cumulative += length_frequencies[value]
        encoder.put(cumulative, length_frequencies[symbol], length_total)
        length_frequencies[symbol] += 22
        length_total += 22
        if length_total > 32768:
            length_total = 0
            for value in range(max_length):
                length_frequencies[value] = (length_frequencies[value] + 1) // 2
                length_total += length_frequencies[value]
    stream = encoder.finish()
    header = (b"U" + len(data).to_bytes(4, "big") +
              bytes((max_length,)))
    return header + primary.to_bytes(4, "big") + stream


def _decode_bwt(blob):
    if len(blob) < 10:
        raise ValueError("truncated transform header")
    original_length = int.from_bytes(blob[1:5], "big")
    max_length = blob[5]
    primary = int.from_bytes(blob[6:10], "big")
    if max_length == 0:
        raise ValueError("invalid transform streams")
    decoder = _Decoder(blob[10:])
    rank_frequencies = [1] * 257
    rank_total = 257
    length_frequencies = [1] * max_length
    length_total = max_length
    target_size = original_length + 1
    lengths = []
    ranks = []
    covered = 0
    while covered < target_size:
        target = decoder.target(rank_total)
        cumulative = 0
        symbol = -1
        for value, frequency in enumerate(rank_frequencies):
            if target < cumulative + frequency:
                symbol = value
                decoder.take(cumulative, frequency, rank_total)
                break
            cumulative += frequency
        if symbol < 0:
            raise ValueError("rank stream escaped its alphabet")
        ranks.append(symbol)
        rank_frequencies[symbol] += 12
        rank_total += 12
        if rank_total > 32768:
            rank_total = 0
            for value in range(257):
                rank_frequencies[value] = (rank_frequencies[value] + 1) // 2
                rank_total += rank_frequencies[value]

        target = decoder.target(length_total)
        cumulative = 0
        symbol = -1
        for value, frequency in enumerate(length_frequencies):
            if target < cumulative + frequency:
                symbol = value
                decoder.take(cumulative, frequency, length_total)
                break
            cumulative += frequency
        if symbol < 0:
            raise ValueError("length stream escaped its alphabet")
        length = symbol + 1
        covered += length
        if covered > target_size:
            raise ValueError("length stream overshot transform")
        lengths.append(length)
        length_frequencies[symbol] += 22
        length_total += 22
        if length_total > 32768:
            length_total = 0
            for value in range(max_length):
                length_frequencies[value] = (length_frequencies[value] + 1) // 2
                length_total += length_frequencies[value]

    used = [False] * 257
    alphabet = []
    for symbol in _PREFERRED + tuple(range(257)):
        if not used[symbol]:
            used[symbol] = True
            alphabet.append(symbol)
    last = []
    for rank, length in zip(ranks, lengths):
        if rank >= len(alphabet):
            raise ValueError("invalid move-to-front rank")
        symbol = alphabet.pop(rank)
        alphabet.insert(0, symbol)
        last.extend([symbol] * length)

    size = original_length + 1
    if len(last) != size or primary >= size:
        raise ValueError("invalid inverse transform")
    counts = [0] * 257
    for symbol in last:
        counts[symbol] += 1
    starts = [0] * 257
    offset = 0
    for symbol in range(257):
        starts[symbol] = offset
        offset += counts[symbol]
    seen = [0] * 257
    links = [0] * size
    for row, symbol in enumerate(last):
        links[row] = starts[symbol] + seen[symbol]
        seen[symbol] += 1
    row = primary
    restored = [0] * size
    for at in range(size - 1, -1, -1):
        restored[at] = last[row]
        row = links[row]
    if restored[-1] != 256:
        raise ValueError("transform sentinel missing")
    return bytes(restored[:-1])


def compress(data: bytes) -> bytes:
    raw = bytes(data)
    transformed = _encode_bwt(raw)
    literal = b"R" + len(raw).to_bytes(4, "big") + raw
    return transformed if len(transformed) < len(literal) else literal


def decompress(blob: bytes) -> bytes:
    if len(blob) < 5:
        raise ValueError("truncated compressed data")
    if blob[0] == 82:
        size = int.from_bytes(blob[1:5], "big")
        raw = blob[5:]
        if len(raw) != size:
            raise ValueError("literal size mismatch")
        return bytes(raw)
    if blob[0] == 85:
        return _decode_bwt(blob)
    raise ValueError("unknown compressed stream")
