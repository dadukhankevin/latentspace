"""Context-adaptive LZ/range codec for the evolutionary compression task."""

FULL = 0xFFFFFFFF
HALF = 1 << 31
Q1 = 1 << 30
Q3 = HALF + Q1
PINIT = 2048


class Encoder:
    def __init__(self):
        self.low = 0
        self.high = FULL
        self.pending = 0
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def _write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def _emit(self, bit):
        self._write(bit)
        while self.pending:
            self._write(1 - bit)
            self.pending -= 1

    def bit(self, prob_zero, bit):
        span = self.high - self.low + 1
        cut = self.low + (span * prob_zero // 4096)
        if bit == 0:
            self.high = cut - 1
        else:
            self.low = cut
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
        while self.nbits:
            self._write(0)
        return bytes(self.out)


class Decoder:
    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.bitpos = 0
        self.low = 0
        self.high = FULL
        self.value = 0
        for _ in range(4):
            self.value = (self.value << 8) | self._byte()

    def _byte(self):
        if self.pos < len(self.data):
            value = self.data[self.pos]
            self.pos += 1
            return value
        return 0

    def _read(self):
        if self.pos >= len(self.data):
            return 0
        bit = (self.data[self.pos] >> (7 - self.bitpos)) & 1
        self.bitpos += 1
        if self.bitpos == 8:
            self.bitpos = 0
            self.pos += 1
        return bit

    def bit(self, prob_zero):
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * 4096 - 1) // span
        cut = self.low + (span * prob_zero // 4096)
        if target < prob_zero:
            bit = 0
            self.high = cut - 1
        else:
            bit = 1
            self.low = cut
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
            self.value = (self.value << 1) | self._read()
        return bit


def _adapt(p, bit):
    if bit:
        p -= p >> 5
    else:
        p += (4096 - p) >> 5
    if p < 1:
        return 1
    if p > 4095:
        return 4095
    return p


class State:
    def __init__(self):
        self.flag = [PINIT, PINIT]
        self.mode = [PINIT, PINIT, PINIT, PINIT]
        self.dcat = [PINIT, PINIT, PINIT]
        self.lcat = [PINIT, PINIT, PINIT]
        self.dval = [PINIT] * 16
        self.lval = [PINIT] * 16
        self.p1 = [PINIT] * (256 * 8)
        # The second-order table is deliberately full-sized: it avoids
        # hash collisions in the repeated prose and source fragments.
        self.p2 = [PINIT] * (65536 * 8)
        self.seen = [0] * 65536
        self.prev1 = 0
        self.prev2 = 0
        self.prev_token = 0
        self.dist_hist = []

    def field_bit(self, coder, table, index, bit=None):
        p = table[index]
        if isinstance(coder, Encoder):
            coder.bit(p, bit)
        else:
            bit = coder.bit(p)
        table[index] = _adapt(p, bit)
        return bit

    def literal_bit(self, coder, bitpos, bit=None):
        ctx = (self.prev2 << 8) | self.prev1
        use2 = self.seen[ctx] >= 2
        table = self.p2 if use2 else self.p1
        index = (ctx << 3) | bitpos if use2 else (self.prev1 << 3) | bitpos
        p = table[index]
        if isinstance(coder, Encoder):
            coder.bit(p, bit)
        else:
            bit = coder.bit(p)
        table[index] = _adapt(p, bit)
        return bit

    def train_byte(self, value):
        ctx = (self.prev2 << 8) | self.prev1
        use2 = self.seen[ctx] >= 2
        i1 = (self.prev1 << 3)
        i2 = (ctx << 3)
        for k in range(8):
            b = (value >> (7 - k)) & 1
            self.p1[i1 + k] = _adapt(self.p1[i1 + k], b)
            self.p2[i2 + k] = _adapt(self.p2[i2 + k], b)
        if self.seen[ctx] < 255:
            self.seen[ctx] += 1
        self.prev2 = self.prev1
        self.prev1 = value

    def move_distance(self, distance):
        if distance in self.dist_hist:
            self.dist_hist.remove(distance)
        self.dist_hist.insert(0, distance)
        del self.dist_hist[4:]


def _put_bits(state, coder, table, value, width):
    for k in range(width - 1, -1, -1):
        state.field_bit(coder, table, width - 1 - k, (value >> k) & 1)


def _get_bits(state, coder, table, width):
    value = 0
    for k in range(width):
        value = (value << 1) | state.field_bit(coder, table, k)
    return value


def _put_distance(state, coder, distance):
    if distance <= 16:
        state.field_bit(coder, state.dcat, 0, 0)
        _put_bits(state, coder, state.dval, distance - 1, 4)
    elif distance <= 256:
        state.field_bit(coder, state.dcat, 0, 1)
        state.field_bit(coder, state.dcat, 1, 0)
        _put_bits(state, coder, state.dval, distance - 1, 8)
    elif distance <= 4096:
        state.field_bit(coder, state.dcat, 0, 1)
        state.field_bit(coder, state.dcat, 1, 1)
        state.field_bit(coder, state.dcat, 2, 0)
        _put_bits(state, coder, state.dval, distance - 1, 12)
    else:
        state.field_bit(coder, state.dcat, 0, 1)
        state.field_bit(coder, state.dcat, 1, 1)
        state.field_bit(coder, state.dcat, 2, 1)
        _put_bits(state, coder, state.dval, distance - 1, 16)


def _get_distance(state, coder):
    if state.field_bit(coder, state.dcat, 0) == 0:
        return _get_bits(state, coder, state.dval, 4) + 1
    if state.field_bit(coder, state.dcat, 1) == 0:
        return _get_bits(state, coder, state.dval, 8) + 1
    if state.field_bit(coder, state.dcat, 2) == 0:
        return _get_bits(state, coder, state.dval, 12) + 1
    return _get_bits(state, coder, state.dval, 16) + 1


def _put_length(state, coder, length):
    value = length - 4
    if length <= 19:
        state.field_bit(coder, state.lcat, 0, 0)
        _put_bits(state, coder, state.lval, value, 4)
    elif length <= 259:
        state.field_bit(coder, state.lcat, 0, 1)
        state.field_bit(coder, state.lcat, 1, 0)
        _put_bits(state, coder, state.lval, value, 8)
    elif length <= 4355:
        state.field_bit(coder, state.lcat, 0, 1)
        state.field_bit(coder, state.lcat, 1, 1)
        state.field_bit(coder, state.lcat, 2, 0)
        _put_bits(state, coder, state.lval, value, 12)
    else:
        state.field_bit(coder, state.lcat, 0, 1)
        state.field_bit(coder, state.lcat, 1, 1)
        state.field_bit(coder, state.lcat, 2, 1)
        _put_bits(state, coder, state.lval, value, 16)


def _get_length(state, coder):
    if state.field_bit(coder, state.lcat, 0) == 0:
        return _get_bits(state, coder, state.lval, 4) + 4
    if state.field_bit(coder, state.lcat, 1) == 0:
        return _get_bits(state, coder, state.lval, 8) + 4
    if state.field_bit(coder, state.lcat, 2) == 0:
        return _get_bits(state, coder, state.lval, 12) + 4
    return _get_bits(state, coder, state.lval, 16) + 4


def _match_cost(distance, length):
    if distance <= 16:
        dcost = 5
    elif distance <= 256:
        dcost = 10
    elif distance <= 4096:
        dcost = 15
    else:
        dcost = 19
    if length <= 19:
        lcost = 5
    elif length <= 259:
        lcost = 10
    elif length <= 4355:
        lcost = 15
    else:
        lcost = 19
    # Four unary mode bits are the conservative price of a new distance.
    return 1 + 4 + dcost + lcost


def _find_matches(data):
    n = len(data)
    buckets = {}
    matches = [[] for _ in range(n)]
    max_chain = 40
    max_length = 4096
    for p in range(n):
        if p + 3 <= n:
            key = (data[p] << 16) | (data[p + 1] << 8) | data[p + 2]
            chain = buckets.get(key)
            if chain:
                best_by_cat = {}
                for q in reversed(chain[-max_chain:]):
                    distance = p - q
                    if distance > 65536:
                        break
                    limit = min(n - p, max_length)
                    length = 3
                    while length < limit and data[q + length] == data[p + length]:
                        length += 1
                    if length >= 4:
                        if distance <= 16:
                            cat = 0
                        elif distance <= 256:
                            cat = 1
                        elif distance <= 4096:
                            cat = 2
                        else:
                            cat = 3
                        old = best_by_cat.get(cat)
                        if old is None or length > old[1]:
                            best_by_cat[cat] = (distance, length)
                matches[p] = list(best_by_cat.values())
            if chain is None:
                buckets[key] = [p]
            else:
                chain.append(p)
                if len(chain) > max_chain * 3:
                    del chain[:-max_chain * 2]
    return matches


def _candidate_lengths(longest):
    if longest <= 0:
        return ()
    # Short lengths allow alignment choices; longer ranges are represented by
    # their ends and by the points where the field width changes.
    vals = set(range(4, min(longest, 48) + 1))
    vals.update((19, 20, 32, 48, 64, 96, 128, 192, 256, 259,
                 260, 384, 512, 768, 1024, 2048, 4095, 4355, longest))
    return tuple(sorted(x for x in vals if 4 <= x <= longest))


def _parse(data):
    n = len(data)
    matches = _find_matches(data)
    inf = 10 ** 12
    cost = [inf] * (n + 1)
    kind = bytearray(n)
    arg_d = [0] * n
    arg_l = [0] * n
    cost[n] = 0
    literal_price = 4.25
    for p in range(n - 1, -1, -1):
        best = literal_price + cost[p + 1]
        bk = 0
        bd = 0
        bl = 0
        for distance, longest in matches[p]:
            for length in _candidate_lengths(longest):
                if p + length <= n:
                    v = _match_cost(distance, length) + cost[p + length]
                    if v < best:
                        best = v
                        bk = 1
                        bd = distance
                        bl = length
        cost[p] = best
        kind[p] = bk
        arg_d[p] = bd
        arg_l[p] = bl
    return kind, arg_d, arg_l


def _encode_literal(state, coder, value):
    for k in range(8):
        b = (value >> (7 - k)) & 1
        state.literal_bit(coder, k, b)
    state.train_byte(value)


def _decode_literal(state, coder):
    value = 0
    for k in range(8):
        value = (value << 1) | state.literal_bit(coder, k)
    state.train_byte(value)
    return value


def _encode_mode(state, coder, distance):
    try:
        index = state.dist_hist.index(distance)
    except ValueError:
        index = 4
    for k in range(index):
        state.field_bit(coder, state.mode, k, 1)
    if index < 4:
        state.field_bit(coder, state.mode, index, 0)
        return
    _put_distance(state, coder, distance)


def _decode_mode(state, coder):
    index = 0
    while index < 4 and state.field_bit(coder, state.mode, index) == 1:
        index += 1
    if index < 4:
        return state.dist_hist[index]
    return _get_distance(state, coder)


def compress(data: bytes) -> bytes:
    if not data:
        return b"\0\0\0\0" + Encoder().finish()
    kind, arg_d, arg_l = _parse(data)
    state = State()
    coder = Encoder()
    p = 0
    while p < len(data):
        is_match = kind[p]
        flag = state.flag[state.prev_token]
        if is_match:
            coder.bit(flag, 1)
            state.flag[state.prev_token] = _adapt(flag, 1)
            distance = arg_d[p]
            length = arg_l[p]
            _encode_mode(state, coder, distance)
            _put_length(state, coder, length)
            for j in range(length):
                state.train_byte(data[p + j])
            state.move_distance(distance)
            state.prev_token = 1
            p += length
        else:
            coder.bit(flag, 0)
            state.flag[state.prev_token] = _adapt(flag, 0)
            _encode_literal(state, coder, data[p])
            state.prev_token = 0
            p += 1
    return len(data).to_bytes(4, "big") + coder.finish()


def decompress(blob: bytes) -> bytes:
    n = int.from_bytes(blob[:4], "big")
    state = State()
    coder = Decoder(blob[4:])
    out = bytearray()
    while len(out) < n:
        flag = state.flag[state.prev_token]
        is_match = coder.bit(flag)
        state.flag[state.prev_token] = _adapt(flag, is_match)
        if not is_match:
            out.append(_decode_literal(state, coder))
            state.prev_token = 0
            continue
        distance = _decode_mode(state, coder)
        length = _get_length(state, coder)
        start = len(out) - distance
        if distance <= 0 or start < 0 or length <= 0 or len(out) + length > n:
            raise ValueError("invalid stream")
        for j in range(length):
            value = out[start + j]
            out.append(value)
            state.train_byte(value)
        state.move_distance(distance)
        state.prev_token = 1
    return bytes(out)
