"""A small exclusion PPM model carried by a binary arithmetic coder."""

TOP = 0xffffffff
MID = 0x80000000
LO = 0x40000000
HI = MID + LO
ROW_CEILING = 1024
BASE_CEILING = 32768


class BitSink:
    def __init__(self):
        self.data = bytearray()
        self.word = 0
        self.used = 0

    def bit(self, value):
        self.word = (self.word << 1) | value
        self.used += 1
        if self.used == 8:
            self.data.append(self.word)
            self.word = 0
            self.used = 0

    def finish(self):
        while self.used:
            self.bit(0)
        return bytes(self.data)


class BitSource:
    def __init__(self, data):
        self.data = data
        self.at = 0

    def bit(self):
        if self.at >= len(self.data) * 8:
            return 0
        b = self.data[self.at >> 3]
        answer = (b >> (7 - (self.at & 7))) & 1
        self.at += 1
        return answer


class ArithmeticOut:
    def __init__(self):
        self.low = 0
        self.high = TOP
        self.follow = 0
        self.bits = BitSink()

    def _send(self, value):
        self.bits.bit(value)
        while self.follow:
            self.bits.bit(value ^ 1)
            self.follow -= 1

    def put(self, start, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (start + width)) // total - 1
        self.low = self.low + (span * start) // total
        while True:
            if self.high < MID:
                self._send(0)
            elif self.low >= MID:
                self._send(1)
                self.low -= MID
                self.high -= MID
            elif self.low >= LO and self.high < HI:
                self.follow += 1
                self.low -= LO
                self.high -= LO
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def close(self):
        self.follow += 1
        self._send(0 if self.low < LO else 1)
        return self.bits.finish()


class ArithmeticIn:
    def __init__(self, data):
        self.low = 0
        self.high = TOP
        self.value = 0
        self.bits = BitSource(data)
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.bit()

    def where(self, total):
        span = self.high - self.low + 1
        return ((self.value - self.low + 1) * total - 1) // span

    def put(self, start, width, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (start + width)) // total - 1
        self.low = self.low + (span * start) // total
        while True:
            if self.high < MID:
                pass
            elif self.low >= MID:
                self.low -= MID
                self.high -= MID
                self.value -= MID
            elif self.low >= LO and self.high < HI:
                self.low -= LO
                self.high -= LO
                self.value -= LO
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.bits.bit()


class ContextModel:
    def __init__(self):
        self.rows = [{}, {}, {}, {}, {}]
        self.base = [1] * 256
        self.base_sum = 256
        self.history = []

    def _key(self, order):
        if len(self.history) < order:
            return None
        return tuple(self.history[-order:])

    def _active(self, row, blocked):
        active = []
        total = 0
        for symbol, amount in row.items():
            if not blocked[symbol]:
                active.append((symbol, amount))
                total += amount
        return active, total

    def _base_active(self, blocked):
        active = []
        total = 0
        for symbol, amount in enumerate(self.base):
            if not blocked[symbol]:
                active.append((symbol, amount))
                total += amount
        return active, total

    def encode(self, coder, symbol):
        blocked = bytearray(256)
        for order in range(4, 0, -1):
            key = self._key(order)
            if key is None:
                continue
            row = self.rows[order].get(key)
            if not row:
                continue
            active, mass = self._active(row, blocked)
            if not active:
                continue
            escape = len(active)
            cursor = 0
            seen = False
            for candidate, amount in active:
                if candidate == symbol:
                    coder.put(cursor, amount, mass + escape)
                    seen = True
                    break
                cursor += amount
            if seen:
                self.learn(symbol)
                return
            coder.put(mass, escape, mass + escape)
            for candidate, _ in active:
                blocked[candidate] = 1

        active, mass = self._base_active(blocked)
        cursor = 0
        for candidate, amount in active:
            if candidate == symbol:
                coder.put(cursor, amount, mass)
                self.learn(symbol)
                return
            cursor += amount
        raise ValueError("no base symbol")

    def decode(self, coder):
        blocked = bytearray(256)
        for order in range(4, 0, -1):
            key = self._key(order)
            if key is None:
                continue
            row = self.rows[order].get(key)
            if not row:
                continue
            active, mass = self._active(row, blocked)
            if not active:
                continue
            escape = len(active)
            target = coder.where(mass + escape)
            if target < mass:
                cursor = 0
                for candidate, amount in active:
                    if target < cursor + amount:
                        coder.put(cursor, amount, mass + escape)
                        self.learn(candidate)
                        return candidate
                    cursor += amount
            else:
                coder.put(mass, escape, mass + escape)
                for candidate, _ in active:
                    blocked[candidate] = 1

        active, mass = self._base_active(blocked)
        target = coder.where(mass)
        cursor = 0
        for candidate, amount in active:
            if target < cursor + amount:
                coder.put(cursor, amount, mass)
                self.learn(candidate)
                return candidate
            cursor += amount
        raise ValueError("no base symbol")

    def learn(self, symbol):
        self.base[symbol] += 1
        self.base_sum += 1
        if self.base_sum >= BASE_CEILING:
            self.base_sum = 0
            for i, amount in enumerate(self.base):
                amount = (amount + 1) // 2
                self.base[i] = amount
                self.base_sum += amount

        for order in range(1, 5):
            key = self._key(order)
            if key is None:
                continue
            row = self.rows[order].get(key)
            if row is None:
                row = {}
                self.rows[order][key] = row
            row[symbol] = row.get(symbol, 0) + 1
            total = 0
            for amount in row.values():
                total += amount
            if total >= ROW_CEILING:
                for candidate, amount in list(row.items()):
                    row[candidate] = (amount + 1) // 2

        self.history.append(symbol)
        if len(self.history) > 4:
            del self.history[0]


def compress(data: bytes) -> bytes:
    model = ContextModel()
    coder = ArithmeticOut()
    for symbol in data:
        model.encode(coder, symbol)
    return len(data).to_bytes(4, "big") + coder.close()


def decompress(blob: bytes) -> bytes:
    size = int.from_bytes(blob[:4], "big")
    model = ContextModel()
    coder = ArithmeticIn(blob[4:])
    answer = bytearray()
    for _ in range(size):
        answer.append(model.decode(coder))
    return bytes(answer)
