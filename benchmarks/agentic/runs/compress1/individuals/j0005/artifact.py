"""A small lossless parser with an adaptive context-coded token stream."""


MASK = 0xFFFFFFFF
TOP = 0xFF000000


class _RangeEncoder:
    def __init__(self):
        self.low = 0
        self.high = MASK
        self.out = bytearray()

    def put(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
        while ((self.low ^ self.high) & TOP) == 0:
            self.out.append(self.high >> 24)
            self.low = (self.low << 8) & MASK
            self.high = ((self.high << 8) | 255) & MASK

    def finish(self):
        for _ in range(4):
            self.out.append(self.low >> 24)
            self.low = (self.low << 8) & MASK
        return bytes(self.out)


class _RangeDecoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.low = 0
        self.high = MASK
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8) | self._read()

    def _read(self):
        if self.pos == len(self.data):
            return 0
        x = self.data[self.pos]
        self.pos += 1
        return x

    def target(self, total):
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def take(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
        while ((self.low ^ self.high) & TOP) == 0:
            self.low = (self.low << 8) & MASK
            self.high = ((self.high << 8) | 255) & MASK
            self.code = ((self.code << 8) & MASK) | self._read()


def _new_model():
    # The final entry is an escape count; the dict contains only seen bytes.
    return [1, {}]


def _model_put(coder, model, symbol):
    esc, counts = model
    old = counts.get(symbol)
    if old is not None:
        cumulative = 0
        for value, frequency in counts.items():
            if value == symbol:
                coder.put(cumulative, frequency, esc + cumulative + frequency +
                         sum(counts.values()) - cumulative - frequency)
                return True
            cumulative += frequency
    total_seen = sum(counts.values())
    coder.put(total_seen, esc, total_seen + esc)
    return False


def _model_get(coder, model):
    esc, counts = model
    total_seen = sum(counts.values())
    total = total_seen + esc
    target = coder.target(total)
    cumulative = 0
    for value, frequency in counts.items():
        if target < cumulative + frequency:
            coder.take(cumulative, frequency, total)
            return value, True
        cumulative += frequency
    coder.take(cumulative, esc, total)
    return None, False


def _update_model(model, symbol):
    esc, counts = model
    old = counts.get(symbol)
    if old is None:
        counts[symbol] = 1
        model[0] = esc + 1
    else:
        counts[symbol] = old + 1


def _put_order_zero(coder, frequencies, total, symbol):
    cumulative = 0
    for value in range(symbol):
        cumulative += frequencies[value]
    coder.put(cumulative, frequencies[symbol], total)


def _get_order_zero(coder, frequencies, total):
    target = coder.target(total)
    cumulative = 0
    for value, frequency in enumerate(frequencies):
        if target < cumulative + frequency:
            coder.take(cumulative, frequency, total)
            return value
        cumulative += frequency
    raise ValueError("arithmetic code escaped the order-zero model")


def _update_order_zero(frequencies, total, symbol):
    frequencies[symbol] += 1
    total += 1
    if total >= 32768:
        total = 0
        for i in range(256):
            frequencies[i] = (frequencies[i] + 1) // 2
            total += frequencies[i]
    return total


def _ppm_encode(data):
    coder = _RangeEncoder()
    order_one = {}
    order_two = {}
    frequencies = [1] * 256
    total_zero = 256
    previous_two = 256
    previous_one = 256

    for symbol in data:
        key_two = (previous_two, previous_one)
        model_two = order_two.get(key_two)
        if model_two is None:
            model_two = _new_model()
            order_two[key_two] = model_two
        if not _model_put(coder, model_two, symbol):
            model_one = order_one.get(previous_one)
            if model_one is None:
                model_one = _new_model()
                order_one[previous_one] = model_one
            if not _model_put(coder, model_one, symbol):
                _put_order_zero(coder, frequencies, total_zero, symbol)

        _update_model(model_two, symbol)
        model_one = order_one.get(previous_one)
        if model_one is None:
            model_one = _new_model()
            order_one[previous_one] = model_one
        _update_model(model_one, symbol)
        total_zero = _update_order_zero(frequencies, total_zero, symbol)
        previous_two, previous_one = previous_one, symbol

    return coder.finish()


def _ppm_decode(data, count):
    coder = _RangeDecoder(data)
    order_one = {}
    order_two = {}
    frequencies = [1] * 256
    total_zero = 256
    previous_two = 256
    previous_one = 256
    output = bytearray()

    for _ in range(count):
        key_two = (previous_two, previous_one)
        model_two = order_two.get(key_two)
        if model_two is None:
            model_two = _new_model()
            order_two[key_two] = model_two
        symbol, present = _model_get(coder, model_two)
        if not present:
            model_one = order_one.get(previous_one)
            if model_one is None:
                model_one = _new_model()
                order_one[previous_one] = model_one
            symbol, present = _model_get(coder, model_one)
            if not present:
                symbol = _get_order_zero(coder, frequencies, total_zero)

        _update_model(model_two, symbol)
        model_one = order_one.get(previous_one)
        if model_one is None:
            model_one = _new_model()
            order_one[previous_one] = model_one
        _update_model(model_one, symbol)
        total_zero = _update_order_zero(frequencies, total_zero, symbol)
        output.append(symbol)
        previous_two, previous_one = previous_one, symbol

    return bytes(output)


def _bwt_forward(data):
    values = list(data) + [256]
    size = len(values)
    suffixes = list(range(size))
    ranks = values[:]
    step = 1
    while step < size:
        suffixes.sort(key=lambda i: (ranks[i], ranks[(i + step) % size]))
        new_ranks = [0] * size
        classes = 0
        previous = None
        for i in suffixes:
            pair = (ranks[i], ranks[(i + step) % size])
            if previous is not None and pair != previous:
                classes += 1
            new_ranks[i] = classes
            previous = pair
        ranks = new_ranks
        if classes == size - 1:
            break
        step <<= 1
    primary = 0
    last = [0] * size
    for row, start in enumerate(suffixes):
        if start == 0:
            primary = row
        last[row] = values[(start - 1) % size]
    return last, primary


def _mtf_ranks(last):
    alphabet = list(range(257))
    ranks = []
    for symbol in last:
        rank = alphabet.index(symbol)
        ranks.append(rank)
        alphabet.pop(rank)
        alphabet.insert(0, symbol)
    return ranks


def _zero_run_encode(ranks):
    encoded = []
    i = 0
    while i < len(ranks):
        rank = ranks[i]
        if rank != 0:
            encoded.append(rank + 1)
            i += 1
            continue
        end = i + 1
        while end < len(ranks) and ranks[end] == 0:
            end += 1
        run = end - i
        value = run - 1
        while True:
            encoded.append(value & 1)
            value >>= 1
            if value == 0:
                break
        i = end
    return encoded


def _zero_run_decode(encoded):
    ranks = []
    i = 0
    while i < len(encoded):
        symbol = encoded[i]
        i += 1
        if symbol >= 2:
            ranks.append(symbol - 1)
            continue
        value = 0
        shift = 0
        while True:
            value |= (symbol & 1) << shift
            shift += 1
            if i == len(encoded) or encoded[i] >= 2:
                break
            symbol = encoded[i]
            i += 1
        ranks.extend([0] * (value + 1))
    return ranks


def _rank_encode(symbols):
    coder = _RangeEncoder()
    frequencies = [1] * 258
    total = 258
    for symbol in symbols:
        cumulative = 0
        for value in range(symbol):
            cumulative += frequencies[value]
        coder.put(cumulative, frequencies[symbol], total)
        frequencies[symbol] += 18
        total += 18
        if total > 32768:
            total = 0
            for i in range(258):
                frequencies[i] = (frequencies[i] + 1) // 2
                total += frequencies[i]
    return coder.finish()


def _rank_decode(data, count):
    coder = _RangeDecoder(data)
    frequencies = [1] * 258
    total = 258
    symbols = []
    for _ in range(count):
        target = coder.target(total)
        cumulative = 0
        symbol = -1
        for value, frequency in enumerate(frequencies):
            if target < cumulative + frequency:
                symbol = value
                coder.take(cumulative, frequency, total)
                break
            cumulative += frequency
        if symbol < 0:
            raise ValueError("arithmetic code escaped the rank model")
        symbols.append(symbol)
        frequencies[symbol] += 18
        total += 18
        if total > 32768:
            total = 0
            for i in range(258):
                frequencies[i] = (frequencies[i] + 1) // 2
                total += frequencies[i]
    return symbols


def _bwt_inverse(last, primary, original_length):
    size = original_length + 1
    counts = [0] * 257
    for symbol in last:
        counts[symbol] += 1
    starts = [0] * 257
    total = 0
    for symbol in range(257):
        starts[symbol] = total
        total += counts[symbol]
    seen = [0] * 257
    links = [0] * size
    for row, symbol in enumerate(last):
        links[row] = starts[symbol] + seen[symbol]
        seen[symbol] += 1
    row = primary
    restored = [0] * size
    for i in range(size - 1, -1, -1):
        restored[i] = last[row]
        row = links[row]
    if restored[-1] != 256:
        raise ValueError("missing transform sentinel")
    return bytes(restored[:-1])


def _bwt_blob(data):
    last, primary = _bwt_forward(data)
    ranks = _zero_run_encode(_mtf_ranks(last))
    coded = _rank_encode(ranks)
    return (b"B" + len(data).to_bytes(4, "big") +
            primary.to_bytes(4, "big") + len(ranks).to_bytes(4, "big") +
            coded)


def _bwt_unblob(blob):
    if len(blob) < 13:
        raise ValueError("truncated transform header")
    original_length = int.from_bytes(blob[1:5], "big")
    primary = int.from_bytes(blob[5:9], "big")
    rank_count = int.from_bytes(blob[9:13], "big")
    encoded = _rank_decode(blob[13:], rank_count)
    ranks = _zero_run_decode(encoded)
    if len(ranks) != original_length + 1:
        raise ValueError("rank stream length mismatch")
    alphabet = list(range(257))
    last = []
    for rank in ranks:
        if rank >= len(alphabet):
            raise ValueError("invalid move-to-front rank")
        symbol = alphabet.pop(rank)
        alphabet.insert(0, symbol)
        last.append(symbol)
    return _bwt_inverse(last, primary, original_length)


def _token_options(data):
    """Find a compact candidate set of prior matches for every position."""
    n = len(data)
    positions = {}
    options = [[] for _ in range(n)]
    sources = [-1] * n
    for i in range(n):
        if i + 3 < n:
            key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
            prior = positions.get(key, ())
            best = 0
            best_source = -1
            # Recent candidates catch local repetitions; the window makes the
            # two-byte distance field sufficient and bounded.
            for p in reversed(prior[-16:]):
                if i - p > 32768:
                    continue
                limit = min(258, n - i)
                length = 0
                while length < limit and data[p + length] == data[i + length]:
                    length += 1
                if length > best:
                    best = length
                    best_source = p
            if best >= 4:
                choices = (4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32,
                           48, 64, 96, 128, 160, 192, 224, best)
                options[i] = sorted(set(x for x in choices if x <= best))
                sources[i] = best_source
        if i + 2 < n:
            key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
            prior = positions.setdefault(key, [])
            prior.append(i)
            if len(prior) > 32:
                del prior[:-32]
    return options, sources


def _make_tokens(data):
    n = len(data)
    options, sources = _token_options(data)
    cost = [0] * (n + 1)
    choices = [0] * n
    choice_sources = [-1] * n
    for i in range(n - 1, -1, -1):
        selected = 0
        best_cost = 2 + cost[i + 1]
        for length in options[i]:
            candidate = 4 + cost[i + length]
            if candidate < best_cost:
                best_cost = candidate
                selected = length
                choice_sources[i] = sources[i]
        cost[i] = best_cost
        choices[i] = selected

    tokens = bytearray()
    i = 0
    while i < n:
        length = choices[i]
        if length == 0:
            tokens.append(0)
            tokens.append(data[i])
            i += 1
            continue
        source = choice_sources[i]
        distance = i - source if source >= 0 else 0
        if distance == 0:
            tokens.append(0)
            tokens.append(data[i])
            i += 1
            continue
        encoded_distance = distance - 1
        tokens.append(1)
        tokens.append(encoded_distance >> 8)
        tokens.append(encoded_distance & 255)
        tokens.append(length - 4)
        i += length
    return bytes(tokens)


def compress(data: bytes) -> bytes:
    raw = bytes(data)
    tokens = _make_tokens(raw)
    coded = _ppm_encode(tokens)
    lz_blob = (b"L" + len(raw).to_bytes(4, "big") +
               len(tokens).to_bytes(4, "big") + coded)
    bwt_blob = _bwt_blob(raw)
    raw_blob = b"R" + len(raw).to_bytes(4, "big") + raw
    return min((lz_blob, bwt_blob, raw_blob), key=len)


def decompress(blob: bytes) -> bytes:
    if not blob:
        raise ValueError("truncated header")
    if blob[0] == 82:
        if len(blob) < 5:
            raise ValueError("truncated raw header")
        original_length = int.from_bytes(blob[1:5], "big")
        raw = blob[5:]
        if len(raw) != original_length:
            raise ValueError("raw length mismatch")
        return bytes(raw)
    if blob[0] == 66:
        return _bwt_unblob(blob)
    if blob[0] != 76 or len(blob) < 9:
        raise ValueError("truncated header")
    original_length = int.from_bytes(blob[1:5], "big")
    token_length = int.from_bytes(blob[5:9], "big")
    tokens = _ppm_decode(blob[9:], token_length)
    output = bytearray()
    i = 0
    while i < len(tokens):
        tag = tokens[i]
        i += 1
        if tag == 0:
            if i >= len(tokens):
                raise ValueError("truncated literal")
            output.append(tokens[i])
            i += 1
        elif tag == 1:
            if i + 3 > len(tokens):
                raise ValueError("truncated match")
            distance = ((tokens[i] << 8) | tokens[i + 1]) + 1
            length = tokens[i + 2] + 4
            i += 3
            if distance > len(output):
                raise ValueError("invalid match distance")
            for _ in range(length):
                output.append(output[-distance])
        else:
            raise ValueError("invalid token tag")
    if len(output) != original_length:
        raise ValueError("decoded length mismatch")
    return bytes(output)
