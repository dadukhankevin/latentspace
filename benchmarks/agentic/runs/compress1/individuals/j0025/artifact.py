"""Sparse order-4 PPM byte coder with exclusion and a range coder."""

MASK = 0xffffffff
HALF = 1 << 31
QUARTER = 1 << 30
THREE_QUARTER = HALF + QUARTER
ROW_LIMIT = 1024
GLOBAL_LIMIT = 1 << 15


class RangeWriter:
    def __init__(self):
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.out = bytearray()
        self.acc = 0
        self.bits = 0

    def put(self, cum, freq, total):
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
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def _write(self, bit):
        self.acc = (self.acc << 1) | bit
        self.bits += 1
        if self.bits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.bits = 0

    def _emit(self, bit):
        self._write(bit)
        while self.pending:
            self._write(1 - bit)
            self.pending -= 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < QUARTER else 1)
        while self.bits:
            self._write(0)
        return bytes(self.out)


class RangeReader:
    def __init__(self, payload):
        self.payload = payload
        self.pos = 0
        self.low = 0
        self.high = MASK
        self.value = 0
        self.bitpos = 0
        for _ in range(32):
            self.value = (self.value << 1) | self._read()

    def _read(self):
        if self.bitpos < len(self.payload) * 8:
            byte = self.payload[self.bitpos >> 3]
            bit = (byte >> (7 - (self.bitpos & 7))) & 1
            self.bitpos += 1
            return bit
        return 0

    def _fill_value(self):
        self.value = (self.value << 1) | self._read()

    def slot(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def take(self, cum, freq, total):
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
            elif self.low >= QUARTER and self.high < THREE_QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self._fill_value()


class PPM:
    def __init__(self):
        self.global_counts = [1] * 256
        self.global_total = 256
        self.tables = [{}, {}, {}, {}, {}]
        self.history = []

    def _row(self, order, key):
        table = self.tables[order]
        row = table.get(key)
        if row is None:
            row = [{}, 0]
            table[key] = row
        return row

    def _context(self, order):
        if order == 0 or len(self.history) < order:
            return None
        return tuple(self.history[-order:])

    @staticmethod
    def _active(row, excluded):
        counts = row[0]
        syms = []
        total = 0
        for sym, count in counts.items():
            if not excluded[sym]:
                syms.append((sym, count))
                total += count
        syms.sort()
        return syms, total

    def _events(self, symbol):
        excluded = [False] * 256
        for order in range(4, 0, -1):
            key = self._context(order)
            if key is None:
                continue
            row = self.tables[order].get(key)
            if row is None or not row[0]:
                continue
            syms, total = self._active(row, excluded)
            if not syms:
                continue
            escape = len(syms)
            if symbol in row[0] and not excluded[symbol]:
                cum = 0
                for sym, count in syms:
                    if sym == symbol:
                        return [(cum, count, total + escape)]
                    cum += count
            yield (total, escape, total + escape)
            for sym, _ in syms:
                excluded[sym] = True

        syms = []
        total = 0
        for sym, count in enumerate(self.global_counts):
            if not excluded[sym]:
                syms.append((sym, count))
                total += count
        cum = 0
        for sym, count in syms:
            if sym == symbol:
                yield (cum, count, total)
                return
            cum += count

    def encode(self, coder, symbol):
        excluded = [False] * 256
        for order in range(4, 0, -1):
            key = self._context(order)
            if key is None:
                continue
            row = self.tables[order].get(key)
            if row is None or not row[0]:
                continue
            syms, total = self._active(row, excluded)
            if not syms:
                continue
            escape = len(syms)
            found = False
            cum = 0
            for sym, count in syms:
                if sym == symbol:
                    coder.put(cum, count, total + escape)
                    found = True
                    break
                cum += count
            if found:
                self.update(symbol)
                return
            coder.put(total, escape, total + escape)
            for sym, _ in syms:
                excluded[sym] = True

        cum = 0
        total = 0
        for sym, count in enumerate(self.global_counts):
            if not excluded[sym]:
                total += count
        for sym, count in enumerate(self.global_counts):
            if excluded[sym]:
                continue
            if sym == symbol:
                coder.put(cum, count, total)
                self.update(symbol)
                return
            cum += count

    def decode(self, coder):
        excluded = [False] * 256
        for order in range(4, 0, -1):
            key = self._context(order)
            if key is None:
                continue
            row = self.tables[order].get(key)
            if row is None or not row[0]:
                continue
            syms, total = self._active(row, excluded)
            if not syms:
                continue
            escape = len(syms)
            full = total + escape
            target = coder.slot(full)
            if target < total:
                cum = 0
                for sym, count in syms:
                    if target < cum + count:
                        coder.take(cum, count, full)
                        self.update(sym)
                        return sym
                    cum += count
            else:
                coder.take(total, escape, full)
                for sym, _ in syms:
                    excluded[sym] = True

        total = 0
        for sym, count in enumerate(self.global_counts):
            if not excluded[sym]:
                total += count
        target = coder.slot(total)
        cum = 0
        for sym, count in enumerate(self.global_counts):
            if excluded[sym]:
                continue
            if target < cum + count:
                coder.take(cum, count, total)
                self.update(sym)
                return sym
            cum += count
        raise ValueError("PPM model has no decodable symbol")

    def update(self, symbol):
        self.global_counts[symbol] += 1
        self.global_total += 1
        if self.global_total >= GLOBAL_LIMIT:
            self.global_total = 0
            for i, count in enumerate(self.global_counts):
                count = (count + 1) // 2
                self.global_counts[i] = count
                self.global_total += count

        for order in range(1, 5):
            key = self._context(order)
            if key is None:
                continue
            row = self._row(order, key)
            counts = row[0]
            counts[symbol] = counts.get(symbol, 0) + 1
            row[1] += 1
            if row[1] >= ROW_LIMIT:
                row[1] = 0
                for sym, count in list(counts.items()):
                    value = (count + 1) // 2
                    counts[sym] = value
                    row[1] += value
        self.history.append(symbol)
        if len(self.history) > 4:
            del self.history[0]


def compress(data: bytes) -> bytes:
    model = PPM()
    coder = RangeWriter()
    for value in data:
        model.encode(coder, value)
    return len(data).to_bytes(4, "big") + coder.finish()


def decompress(blob: bytes) -> bytes:
    size = int.from_bytes(blob[:4], "big")
    model = PPM()
    coder = RangeReader(blob[4:])
    out = bytearray()
    for _ in range(size):
        out.append(model.decode(coder))
    return bytes(out)
