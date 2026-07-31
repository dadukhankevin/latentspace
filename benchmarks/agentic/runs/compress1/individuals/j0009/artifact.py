"""A small lossless LZ coder with context-coded literal payloads."""

import math


def _distance_bits(distance):
    if distance <= 15:
        return 5
    if distance <= 255:
        return 10
    if distance <= 4095:
        return 15
    return 19


def _length_bits(length):
    value = length - 2
    q = value.bit_length() - 1
    return 2 * q + 1


def _huffman(data):
    weights = [1] * 256
    for value in data:
        weights[value] += 1
    node_weight = weights[:]
    left = [-1] * 256
    right = [-1] * 256
    active = list(range(256))
    while len(active) > 1:
        first = 0
        second = 1
        if node_weight[active[second]] < node_weight[active[first]]:
            first, second = second, first
        for p in range(2, len(active)):
            w = node_weight[active[p]]
            if w < node_weight[active[first]]:
                second = first
                first = p
            elif w < node_weight[active[second]]:
                second = p
        a = active[first]
        del active[first]
        if second > first:
            second -= 1
        b = active[second]
        del active[second]
        node_weight.append(node_weight[a] + node_weight[b])
        left.append(a)
        right.append(b)
        active.append(len(node_weight) - 1)
    root = active[0]
    lengths = [0] * 256
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node < 256:
            lengths[node] = depth if depth else 1
        else:
            stack.append((left[node], depth + 1))
            stack.append((right[node], depth + 1))
    maximum = max(lengths)
    count = [0] * (maximum + 1)
    for length in lengths:
        count[length] += 1
    next_code = [0] * (maximum + 1)
    code = 0
    for length in range(1, maximum + 1):
        code = (code + count[length - 1]) << 1
        next_code[length] = code
    codes = [0] * 256
    for symbol in range(256):
        length = lengths[symbol]
        codes[symbol] = next_code[length]
        next_code[length] += 1
    return lengths, codes


def _huffman_table(lengths):
    maximum = max(lengths)
    count = [0] * (maximum + 1)
    for length in lengths:
        count[length] += 1
    next_code = [0] * (maximum + 1)
    code = 0
    for length in range(1, maximum + 1):
        code = (code + count[length - 1]) << 1
        next_code[length] = code
    table = {}
    for symbol, length in enumerate(lengths):
        table[(length, next_code[length])] = symbol
        next_code[length] += 1
    return table, maximum


def _new_models():
    return [[1] * 256 for _ in range(257)], [256] * 257


def _update_model(models, totals, context, value):
    row = models[context]
    row[value] += 1
    totals[context] += 1
    if totals[context] > 16384:
        total = 0
        for i in range(256):
            row[i] = (row[i] + 1) >> 1
            total += row[i]
        totals[context] = total


def _literal_costs(data):
    models, totals = _new_models()
    costs = []
    context = 256
    for value in data:
        total = totals[context]
        frequency = models[context][value]
        costs.append(max(1, int(math.log2(total / frequency) * 256 + 0.5)))
        _update_model(models, totals, context, value)
        context = value
    return costs


def _match_length(data, here, there):
    limit = len(data) - here
    k = 0
    while k < limit and data[here + k] == data[there + k]:
        k += 1
    return k


def _choices(data):
    """Return a deliberately locality-biased set of (distance, length)."""
    n = len(data)
    buckets = {}
    for i in range(n - 2):
        key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        buckets.setdefault(key, []).append(i)

    choices = [[] for _ in range(n)]
    for i in range(n - 2):
        key = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        positions = buckets[key]
        upto = len(positions) - 1
        recent_start = max(0, upto - 11)
        selected = positions[recent_start:upto]

        # A few older representatives retain long-distance repetitions
        # without making the dynamic program quadratic on common trigrams.
        older = positions[:recent_start]
        if older:
            probes = [0, len(older) - 1]
            step = len(older) // 3
            if step:
                probes.extend([step, 2 * step])
            seen = set()
            for p in probes:
                if 0 <= p < len(older) and p not in seen:
                    selected.append(older[p])
                    seen.add(p)

        best_for_distance = {}
        for there in reversed(selected):
            distance = i - there
            if distance <= 0:
                continue
            length = _match_length(data, i, there)
            if length >= 3:
                previous = best_for_distance.get(distance, 0)
                if length > previous:
                    best_for_distance[distance] = length
        choices[i] = list(best_for_distance.items())
    return choices


def _plan(data, literal_costs):
    n = len(data)
    matches = _choices(data)
    cost = [0] * (n + 1)
    kind = bytearray(n)
    arg = [0] * n
    cost[n] = 0

    for i in range(n - 1, -1, -1):
        best = 256 + literal_costs[i] + cost[i + 1]
        best_kind = 0
        best_arg = 0
        for distance, maximum in matches[i]:
            maximum = min(maximum, n - i)
            fixed = (1 + _distance_bits(distance)) * 256
            for length in range(3, maximum + 1):
                candidate = fixed + _length_bits(length) * 256 + cost[i + length]
                if candidate < best:
                    best = candidate
                    best_kind = 1
                    best_arg = (distance << 20) | length
        cost[i] = best
        kind[i] = best_kind
        arg[i] = best_arg
    return kind, arg


class _Writer:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.bits = 0

    def put(self, value, width):
        self.acc = (self.acc << width) | value
        self.bits += width
        while self.bits >= 8:
            self.bits -= 8
            self.out.append((self.acc >> self.bits) & 255)
            if self.bits:
                self.acc &= (1 << self.bits) - 1
            else:
                self.acc = 0

    def finish(self):
        if self.bits:
            self.out.append((self.acc << (8 - self.bits)) & 255)
        return bytes(self.out)


class _Reader:
    def __init__(self, blob):
        self.blob = blob
        self.pos = 0
        self.acc = 0
        self.bits = 0

    def get(self, width):
        while self.bits < width:
            if self.pos < len(self.blob):
                self.acc = (self.acc << 8) | self.blob[self.pos]
            else:
                self.acc <<= 8
            self.pos += 1
            self.bits += 8
        self.bits -= width
        value = self.acc >> self.bits
        if self.bits:
            self.acc &= (1 << self.bits) - 1
        else:
            self.acc = 0
        return value


class _RangeEncoder:
    def __init__(self, writer):
        self.writer = writer
        self.low = 0
        self.high = 0xFFFFFFFF
        self.pending = 0

    def _emit(self, bit):
        self.writer.put(bit, 1)
        while self.pending:
            self.writer.put(1 - bit, 1)
            self.pending -= 1

    def symbol(self, value, models, totals, context):
        row = models[context]
        cumulative = 0
        for i in range(value):
            cumulative += row[i]
        frequency = row[value]
        span = self.high - self.low + 1
        self.high = self.low + span * (cumulative + frequency) // totals[context] - 1
        self.low = self.low + span * cumulative // totals[context]
        while True:
            if self.high < 0x80000000:
                self._emit(0)
            elif self.low >= 0x80000000:
                self._emit(1)
                self.low -= 0x80000000
                self.high -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.pending += 1
                self.low -= 0x40000000
                self.high -= 0x40000000
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.low &= 0xFFFFFFFF
            self.high &= 0xFFFFFFFF

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < 0x40000000 else 1)
        return self.writer.finish()


class _RangeDecoder:
    def __init__(self, reader):
        self.reader = reader
        self.low = 0
        self.high = 0xFFFFFFFF
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.reader.get(1)

    def symbol(self, models, totals, context):
        row = models[context]
        total = totals[context]
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // span
        cumulative = 0
        value = 0
        while cumulative + row[value] <= target:
            cumulative += row[value]
            value += 1
        frequency = row[value]
        self.high = self.low + span * (cumulative + frequency) // total - 1
        self.low = self.low + span * cumulative // total
        while True:
            if self.high < 0x80000000:
                pass
            elif self.low >= 0x80000000:
                self.low -= 0x80000000
                self.high -= 0x80000000
                self.value -= 0x80000000
            elif self.low >= 0x40000000 and self.high < 0xC0000000:
                self.low -= 0x40000000
                self.high -= 0x40000000
                self.value -= 0x40000000
            else:
                break
            self.low = (self.low << 1) & 0xFFFFFFFF
            self.high = ((self.high << 1) | 1) & 0xFFFFFFFF
            self.value = ((self.value << 1) | self.reader.get(1)) & 0xFFFFFFFF
        return value


def _put_distance(writer, distance):
    if distance <= 15:
        writer.put(0, 1)
        writer.put(distance - 1, 4)
    elif distance <= 255:
        writer.put(2, 2)
        writer.put(distance - 16, 8)
    elif distance <= 4095:
        writer.put(6, 3)
        writer.put(distance - 256, 12)
    else:
        writer.put(7, 3)
        writer.put(distance - 4096, 16)


def _get_distance(reader):
    first = reader.get(1)
    if first == 0:
        return reader.get(4) + 1
    second = reader.get(1)
    if second == 0:
        return reader.get(8) + 16
    third = reader.get(1)
    if third == 0:
        return reader.get(12) + 256
    return reader.get(16) + 4096


def _put_length(writer, length):
    value = length - 2
    q = value.bit_length() - 1
    writer.put(0, q)
    writer.put(value, q + 1)


def _get_length(reader):
    q = 0
    while reader.get(1) == 0:
        q += 1
    return reader.get(q) + (1 << q) + 2 if q else 3


def compress(data: bytes) -> bytes:
    n = len(data)
    if n > 0xFFFFFFFF:
        raise ValueError("input too large")
    kind, arg = _plan(data, _literal_costs(data))
    control = _Writer()
    literals = _Writer()
    arithmetic = _RangeEncoder(literals)
    models, totals = _new_models()
    context = 256
    i = 0
    while i < n:
        if kind[i] == 0:
            control.put(0, 1)
            value = data[i]
            arithmetic.symbol(value, models, totals, context)
            _update_model(models, totals, context, value)
            context = value
            i += 1
        else:
            packed = arg[i]
            length = packed & ((1 << 20) - 1)
            distance = packed >> 20
            control.put(1, 1)
            _put_distance(control, distance)
            _put_length(control, length)
            for p in range(i, i + length):
                value = data[p]
                _update_model(models, totals, context, value)
                context = value
            i += length
    control_blob = control.finish()
    literal_blob = arithmetic.finish()
    return (n.to_bytes(4, "big") + len(control_blob).to_bytes(4, "big") +
            control_blob + literal_blob)


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    control_length = int.from_bytes(blob[4:8], "big")
    control = _Reader(blob[8:8 + control_length])
    arithmetic = _RangeDecoder(_Reader(blob[8 + control_length:]))
    models, totals = _new_models()
    context = 256
    out = bytearray()
    while len(out) < n:
        if control.get(1) == 0:
            value = arithmetic.symbol(models, totals, context)
            out.append(value)
            _update_model(models, totals, context, value)
            context = value
        else:
            distance = _get_distance(control)
            length = _get_length(control)
            start = len(out) - distance
            if start < 0:
                raise ValueError("invalid back-reference")
            for _ in range(length):
                value = out[start]
                out.append(value)
                _update_model(models, totals, context, value)
                context = value
                start += 1
    if len(out) != n:
        raise ValueError("invalid stream length")
    return bytes(out)
