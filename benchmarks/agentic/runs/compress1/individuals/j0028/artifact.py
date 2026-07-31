"""BWT/MTF zero-run tokens coded by an exclusion PPM range coder."""

MASK = 0xFFFFFFFF
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTER = HALF + QUARTER

TOKEN_COUNT = 258
MAX_ORDER = 4
ROW_LIMIT = 1152
GLOBAL_LIMIT = 3072


class _BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.value = 0
        self.used = 0

    def put(self, bit):
        self.value = (self.value << 1) | bit
        self.used += 1
        if self.used == 8:
            self.data.append(self.value)
            self.value = 0
            self.used = 0

    def finish(self):
        if self.used:
            self.value <<= 8 - self.used
            self.data.append(self.value)
        return bytes(self.data)


class _BitReader:
    def __init__(self, data):
        self.data = data
        self.position = 0

    def get(self):
        if self.position >= len(self.data) * 8:
            self.position += 1
            return 0
        byte = self.data[self.position >> 3]
        bit = (byte >> (7 - (self.position & 7))) & 1
        self.position += 1
        return bit


class _ArithmeticWriter:
    def __init__(self):
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.bits = _BitWriter()

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(1 - bit)
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
        return self.bits.finish()


class _ArithmeticReader:
    def __init__(self, data):
        self.bits = _BitReader(data)
        self.low = 0
        self.high = MASK
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.get()

    def slot(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def decode(self, cumulative, frequency, total):
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


class _PPM:
    """Sparse order-four PPM with exclusion and an order-zero floor."""

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.global_counts = [1] * alphabet
        self.global_total = alphabet
        self.rows = [{}, {}, {}, {}, {}]
        self.history = []

    def _key(self, order):
        if len(self.history) < order:
            return None
        return tuple(self.history[-order:])

    @staticmethod
    def _active(row, excluded):
        active = []
        total = 0
        for symbol, count in row.items():
            if symbol not in excluded:
                active.append((symbol, count))
                total += count
        return active, total

    def _encode_symbol(self, coder, symbol):
        excluded = set()
        for order in range(MAX_ORDER, 0, -1):
            key = self._key(order)
            if key is None:
                continue
            row = self.rows[order].get(key)
            if not row:
                continue
            active, total = self._active(row, excluded)
            if not active:
                continue
            escape = len(active)
            cumulative = 0
            found = False
            for candidate, count in active:
                if candidate == symbol:
                    coder.encode(cumulative, count, total + escape)
                    found = True
                    break
                cumulative += count
            if found:
                self._update(symbol)
                return
            coder.encode(total, escape, total + escape)
            excluded.update(candidate for candidate, _ in active)

        total = 0
        for candidate, count in enumerate(self.global_counts):
            if candidate not in excluded:
                total += count
        if total <= 0:
            excluded.clear()
            total = sum(self.global_counts)
        cumulative = 0
        for candidate, count in enumerate(self.global_counts):
            if candidate in excluded:
                continue
            if candidate == symbol:
                coder.encode(cumulative, count, total)
                self._update(symbol)
                return
            cumulative += count
        raise ValueError("PPM encoder lost the target symbol")

    def encode(self, coder, symbol):
        self._encode_symbol(coder, symbol)

    def decode(self, coder):
        excluded = set()
        for order in range(MAX_ORDER, 0, -1):
            key = self._key(order)
            if key is None:
                continue
            row = self.rows[order].get(key)
            if not row:
                continue
            active, total = self._active(row, excluded)
            if not active:
                continue
            escape = len(active)
            full = total + escape
            target = coder.slot(full)
            if target < total:
                cumulative = 0
                for symbol, count in active:
                    if target < cumulative + count:
                        coder.decode(cumulative, count, full)
                        self._update(symbol)
                        return symbol
                    cumulative += count
                raise ValueError("PPM row lookup failed")
            coder.decode(total, escape, full)
            excluded.update(symbol for symbol, _ in active)

        total = 0
        for symbol, count in enumerate(self.global_counts):
            if symbol not in excluded:
                total += count
        if total <= 0:
            excluded.clear()
            total = sum(self.global_counts)
        target = coder.slot(total)
        cumulative = 0
        for symbol, count in enumerate(self.global_counts):
            if symbol in excluded:
                continue
            if target < cumulative + count:
                coder.decode(cumulative, count, total)
                self._update(symbol)
                return symbol
            cumulative += count
        raise ValueError("PPM decoder has no available symbol")

    def _update(self, symbol):
        self.global_counts[symbol] += 1
        self.global_total += 1
        if self.global_total >= GLOBAL_LIMIT:
            total = 0
            for index, count in enumerate(self.global_counts):
                reduced = (count + 1) // 2
                self.global_counts[index] = reduced
                total += reduced
            self.global_total = total

        for order in range(1, MAX_ORDER + 1):
            key = self._key(order)
            if key is None:
                continue
            table = self.rows[order]
            row = table.get(key)
            if row is None:
                row = {}
                table[key] = row
            row[symbol] = row.get(symbol, 0) + 1
            if sum(row.values()) >= ROW_LIMIT:
                for candidate, count in tuple(row.items()):
                    reduced = (count + 1) // 2
                    if reduced:
                        row[candidate] = reduced
                    else:
                        del row[candidate]

        self.history.append(symbol)
        if len(self.history) > MAX_ORDER:
            del self.history[0]


def _bwt(data):
    """Return the last column and the row containing the original rotation."""
    n = len(data)
    if n == 0:
        return b"", 0
    order = list(range(n))
    rank = list(data)
    step = 1
    while step < n:
        order.sort(key=lambda start: (rank[start], rank[(start + step) % n]))
        new_rank = [0] * n
        classes = 0
        old_key = None
        for position, start in enumerate(order):
            key = (rank[start], rank[(start + step) % n])
            if position and key != old_key:
                classes += 1
            new_rank[start] = classes
            old_key = key
        rank = new_rank
        if classes == n - 1:
            break
        step <<= 1
    last = bytearray(n)
    for row, start in enumerate(order):
        last[row] = data[(start - 1) % n]
    return bytes(last), order.index(0)


def _inverse_bwt(last, primary):
    n = len(last)
    if n == 0:
        return b""
    counts = [0] * 256
    occurrence = [0] * n
    for position, symbol in enumerate(last):
        occurrence[position] = counts[symbol]
        counts[symbol] += 1
    starts = [0] * 256
    offset = 0
    for symbol, count in enumerate(counts):
        starts[symbol] = offset
        offset += count
    restored = bytearray(n)
    row = primary
    for position in range(n - 1, -1, -1):
        symbol = last[row]
        restored[position] = symbol
        row = starts[symbol] + occurrence[row]
    return bytes(restored)


def _mtf_encode(last):
    alphabet = list(range(256))
    ranks = []
    for symbol in last:
        rank = alphabet.index(symbol)
        ranks.append(rank)
        alphabet.pop(rank)
        alphabet.insert(0, symbol)
    return ranks


def _mtf_decode(ranks):
    alphabet = list(range(256))
    last = bytearray()
    for rank in ranks:
        symbol = alphabet[rank]
        last.append(symbol)
        alphabet.pop(rank)
        alphabet.insert(0, symbol)
    return bytes(last)


def _tokenize(ranks):
    """Turn zero runs into least-significant-first bijective binary digits."""
    tokens = []
    position = 0
    while position < len(ranks):
        rank = ranks[position]
        if rank:
            tokens.append(rank + 1)
            position += 1
            continue
        end = position + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        value = end - position
        while value:
            digit = (value - 1) & 1
            tokens.append(digit)
            value = (value - 1) >> 1
        position = end
    tokens.append(257)
    return tokens


def _detokenize(tokens):
    ranks = []
    position = 0
    while position < len(tokens):
        token = tokens[position]
        if token == 257:
            return ranks
        if token < 2:
            run = 0
            place = 1
            while token < 2:
                run += (token + 1) * place
                place <<= 1
                position += 1
                if position >= len(tokens):
                    raise ValueError("unterminated zero run")
                token = tokens[position]
            ranks.extend([0] * run)
            if token == 257:
                return ranks
        if token < 2 or token > 256:
            raise ValueError("invalid transformed token")
        ranks.append(token - 1)
        position += 1
    raise ValueError("missing token EOF")


def _encode_stream(values, alphabet):
    writer = _ArithmeticWriter()
    model = _PPM(alphabet)
    for value in values:
        model.encode(writer, value)
    return writer.finish()


def _decode_stream(payload, alphabet, count=None):
    reader = _ArithmeticReader(payload)
    model = _PPM(alphabet)
    values = []
    if count is None:
        while True:
            value = model.decode(reader)
            values.append(value)
            if value == 257:
                return values
    for _ in range(count):
        values.append(model.decode(reader))
    return values


def compress(data: bytes) -> bytes:
    if not data:
        return b"\x00" * 4

    # The raw candidate keeps the strong local phrase model available.
    raw_payload = _encode_stream(data, 256)
    raw_blob = len(data).to_bytes(4, "big") + raw_payload

    # The second candidate uses the global rank clustering and compact runs.
    last, primary = _bwt(data)
    ranks = _mtf_encode(last)
    tokens = _tokenize(ranks)
    transformed_payload = _encode_stream(tokens, 258)
    transformed_blob = ((len(data) | 0x80000000).to_bytes(4, "big")
                        + primary.to_bytes(4, "big")
                        + transformed_payload)
    return raw_blob if len(raw_blob) <= len(transformed_blob) else transformed_blob


def decompress(blob: bytes) -> bytes:
    if len(blob) < 4:
        raise ValueError("truncated header")
    tag = int.from_bytes(blob[:4], "big")
    transformed = bool(tag & 0x80000000)
    size = tag & 0x7FFFFFFF
    if size == 0:
        return b""
    if not transformed:
        return bytes(_decode_stream(blob[4:], 256, count=size))
    if len(blob) < 8:
        raise ValueError("truncated transformed header")
    primary = int.from_bytes(blob[4:8], "big")
    tokens = _decode_stream(blob[8:], 258)
    ranks = _detokenize(tokens)
    if len(ranks) != size:
        raise ValueError("transformed length mismatch")
    last = _mtf_decode(ranks)
    return _inverse_bwt(last, primary)
