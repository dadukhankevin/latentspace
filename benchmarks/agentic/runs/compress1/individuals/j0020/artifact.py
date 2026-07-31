"""Sparse order-four PPM byte coder with adaptive arithmetic coding."""

TOP = 0xFFFFFFFF
HALF = 0x80000000
Q1 = 0x40000000
Q3 = 0xC0000000
MAX_ORDER = 4


class _Bits:
    def __init__(self, raw=b""):
        self.buf = bytearray(raw)
        self.acc = 0
        self.used = 0
        self.pos = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.used += 1
        if self.used == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.used = 0

    def get(self):
        if self.pos >= len(self.buf) * 8:
            return 0
        byte = self.buf[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def finish(self):
        while self.used:
            self.put(0)


class _Arithmetic:
    def __init__(self, raw=b""):
        self.bits = _Bits(raw)
        self.low = 0
        self.high = TOP
        self.pending = 0
        self.value = 0

    def start_decode(self):
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.get()

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(bit ^ 1)
            self.pending -= 1

    def encode(self, cum, freq, total):
        span = self.high - self.low + 1
        self.high = self.low + span * (cum + freq) // total - 1
        self.low += span * cum // total
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

    def decode_target(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def decode_interval(self, cum, freq, total):
        span = self.high - self.low + 1
        self.high = self.low + span * (cum + freq) // total - 1
        self.low += span * cum // total
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


def _new_row():
    # Slots 0..255 are counts, 256 is total, 257 is number of symbols seen.
    return [0] * 258


def _initial_global():
    row = [1] * 256 + [256, 256]
    return row


class _PPM:
    def __init__(self):
        self.rows = [{} for _ in range(MAX_ORDER + 1)]
        self.rows[0][0] = _initial_global()
        self.history = 0
        self.length = 0

    def key(self, order):
        if order == 0:
            return 0
        return self.history & ((1 << (order * 8)) - 1)

    def row(self, order):
        return self.rows[order].get(self.key(order))

    def advance(self, symbol):
        # Update every context that preceded this symbol.
        self.rows[0][0][symbol] += 1
        self.rows[0][0][256] += 1
        for order in range(1, MAX_ORDER + 1):
            if self.length < order:
                continue
            key = self.key(order)
            row = self.rows[order].get(key)
            if row is None:
                row = _new_row()
                self.rows[order][key] = row
            if row[symbol] == 0:
                row[257] += 1
            row[symbol] += 1
            row[256] += 1
        self.history = ((self.history << 8) | symbol) & 0xFFFFFFFF
        self.length += 1

    @staticmethod
    def _available(row, excluded):
        total = row[256]
        distinct = row[257]
        if excluded:
            for symbol in range(256):
                if (excluded >> symbol) & 1 and row[symbol]:
                    total -= row[symbol]
                    distinct -= 1
        return total, distinct

    @staticmethod
    def _prefix(row, symbol, excluded):
        cum = sum(row[:symbol])
        if excluded:
            for prior in range(symbol):
                if (excluded >> prior) & 1:
                    cum -= row[prior]
        return cum

    def encode_symbol(self, coder, symbol):
        excluded = 0
        for order in range(MAX_ORDER, 0, -1):
            if self.length < order:
                continue
            row = self.row(order)
            if row is None:
                continue
            known, distinct = self._available(row, excluded)
            if distinct == 0:
                continue
            total = known + distinct
            count = row[symbol]
            if count and not ((excluded >> symbol) & 1):
                coder.encode(self._prefix(row, symbol, excluded), count, total)
                self.advance(symbol)
                return
            coder.encode(known, distinct, total)
            for seen in range(256):
                if row[seen]:
                    excluded |= 1 << seen

        row = self.rows[0][0]
        known = 0
        for seen in range(256):
            if not ((excluded >> seen) & 1):
                known += row[seen]
        coder.encode(self._prefix(row, symbol, excluded), row[symbol], known)
        self.advance(symbol)

    def decode_symbol(self, coder):
        excluded = 0
        for order in range(MAX_ORDER, 0, -1):
            if self.length < order:
                continue
            row = self.row(order)
            if row is None:
                continue
            known, distinct = self._available(row, excluded)
            if distinct == 0:
                continue
            total = known + distinct
            target = coder.decode_target(total)
            if target >= known:
                coder.decode_interval(known, distinct, total)
                for seen in range(256):
                    if row[seen]:
                        excluded |= 1 << seen
                continue

            left = target
            cum = 0
            symbol = 0
            for symbol in range(256):
                if (excluded >> symbol) & 1:
                    continue
                count = row[symbol]
                if count and left < count:
                    break
                cum += count
                left -= count
            coder.decode_interval(cum, row[symbol], total)
            self.advance(symbol)
            return symbol

        row = self.rows[0][0]
        known = 0
        for symbol in range(256):
            if not ((excluded >> symbol) & 1):
                known += row[symbol]
        target = coder.decode_target(known)
        left = target
        cum = 0
        for symbol in range(256):
            if (excluded >> symbol) & 1:
                continue
            count = row[symbol]
            if left < count:
                break
            cum += count
            left -= count
        coder.decode_interval(cum, row[symbol], known)
        self.advance(symbol)
        return symbol


def compress(data: bytes) -> bytes:
    coder = _Arithmetic()
    model = _PPM()
    for symbol in data:
        model.encode_symbol(coder, symbol)
    coder.pending += 1
    coder._emit(0 if coder.low < Q1 else 1)
    coder.bits.finish()
    return len(data).to_bytes(4, "big") + bytes(coder.bits.buf)


def decompress(blob: bytes) -> bytes:
    size = int.from_bytes(blob[:4], "big")
    coder = _Arithmetic(blob[4:])
    coder.start_decode()
    model = _PPM()
    out = bytearray()
    for _ in range(size):
        out.append(model.decode_symbol(coder))
    return bytes(out)
