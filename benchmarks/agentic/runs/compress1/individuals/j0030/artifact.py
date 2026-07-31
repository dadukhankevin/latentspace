"""A small deterministic PPM4 byte coder with an integer arithmetic stream."""

_FULL = 0xFFFFFFFF
_HALF = 0x80000000
_Q1 = 0x40000000
_Q3 = 0xC0000000
_ESC = 256
_EOF = 257
_MAX_ORDER = 4
_ROW_LIMIT = 1 << 15


class _BitsOut:
    def __init__(self):
        self.buf = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, bit):
        self.acc = (self.acc << 1) | bit
        self.n += 1
        if self.n == 8:
            self.buf.append(self.acc)
            self.acc = 0
            self.n = 0

    def finish(self):
        while self.n:
            self.put(0)
        return bytes(self.buf)


class _BitsIn:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get(self):
        if self.pos >= len(self.data) * 8:
            return 0
        b = self.data[self.pos >> 3]
        bit = (b >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit


class _Encoder:
    def __init__(self):
        self.bits = _BitsOut()
        self.low = 0
        self.high = _FULL
        self.pending = 0

    def _emit(self, bit):
        self.bits.put(bit)
        while self.pending:
            self.bits.put(1 - bit)
            self.pending -= 1

    def interval(self, cumulative, frequency, total):
        span = self.high - self.low + 1
        self.high = self.low + (span * (cumulative + frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _Q1 and self.high < _Q3:
                self.pending += 1
                self.low -= _Q1
                self.high -= _Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < _Q1 else 1)
        return self.bits.finish()


class _Decoder:
    def __init__(self, data):
        self.bits = _BitsIn(data)
        self.low = 0
        self.high = _FULL
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self.bits.get()

    def choose(self, choices, total):
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // span
        cumulative = 0
        selected = None
        selected_frequency = 0
        for symbol, frequency in choices:
            if target < cumulative + frequency:
                selected = symbol
                selected_frequency = frequency
                break
            cumulative += frequency
        if selected is None:
            raise ValueError("arithmetic stream selected no symbol")
        self.high = self.low + (span * (cumulative + selected_frequency) // total) - 1
        self.low = self.low + (span * cumulative // total)
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.value -= _HALF
            elif self.low >= _Q1 and self.high < _Q3:
                self.low -= _Q1
                self.high -= _Q1
                self.value -= _Q1
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self.bits.get()
        return selected


class _PPM:
    def __init__(self):
        # All bytes remain possible. Printable characters and common text
        # whitespace get a modest fixed head start, not a future-data peek.
        self.order0 = [1] * (_EOF + 1)
        for symbol in range(32, 127):
            self.order0[symbol] = 64
        for symbol in (9, 10, 13):
            self.order0[symbol] = 64
        self.order0_total = sum(self.order0)
        self.rows = [{}, {}, {}, {}, {}]

    @staticmethod
    def _choices(row, excluded):
        choices = []
        for symbol, frequency in row.items():
            if symbol != _ESC and symbol not in excluded and frequency:
                choices.append((symbol, frequency))
        return choices

    @staticmethod
    def _escape(order, row):
        distinct = len(row) - 1
        if distinct < 1:
            distinct = 1
        return distinct

    def row_for(self, order, history):
        key = bytes(history[-order:])
        return self.rows[order].get(key)

    def update(self, history, symbol):
        self.order0[symbol] += 1
        self.order0_total += 1
        if self.order0_total > _ROW_LIMIT:
            total = 0
            for index, frequency in enumerate(self.order0):
                frequency = (frequency + 1) >> 1
                if not frequency:
                    frequency = 1
                self.order0[index] = frequency
                total += frequency
            self.order0_total = total

        limit = min(_MAX_ORDER, len(history))
        for order in range(1, limit + 1):
            key = bytes(history[-order:])
            row = self.rows[order].get(key)
            if row is None:
                row = {_ESC: 0}
                self.rows[order][key] = row
            row[symbol] = row.get(symbol, 0) + 1
            row[_ESC] += 1
            if row[_ESC] > _ROW_LIMIT:
                total = 0
                for old_symbol in list(row):
                    if old_symbol == _ESC:
                        continue
                    frequency = (row[old_symbol] + 1) >> 1
                    if frequency:
                        row[old_symbol] = frequency
                        total += frequency
                    else:
                        del row[old_symbol]
                row[_ESC] = total


def _encode_with_total(coder, choices, selected):
    cumulative = 0
    total = sum(frequency for _, frequency in choices)
    for symbol, frequency in choices:
        if symbol == selected:
            coder.interval(cumulative, frequency, total)
            return
        cumulative += frequency
    raise ValueError("symbol is absent from model")


def compress(data: bytes) -> bytes:
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    data = bytes(data)
    model = _PPM()
    coder = _Encoder()
    history = bytearray()

    for symbol in data:
        excluded = set()
        encoded = False
        for order in range(min(_MAX_ORDER, len(history)), 0, -1):
            row = model.row_for(order, history)
            if row is None:
                continue
            choices = model._choices(row, excluded)
            if not choices:
                continue
            if symbol in row and symbol not in excluded and row[symbol]:
                choices.append((_ESC, model._escape(order, row)))
                _encode_with_total(coder, choices, symbol)
                encoded = True
                break
            choices.append((_ESC, model._escape(order, row)))
            _encode_with_total(coder, choices, _ESC)
            for represented, frequency in row.items():
                if represented != _ESC and frequency:
                    excluded.add(represented)

        if not encoded:
            choices = [(symbol0, frequency) for symbol0, frequency in
                       enumerate(model.order0) if symbol0 not in excluded]
            _encode_with_total(coder, choices, symbol)

        model.update(history, symbol)
        history.append(symbol)

    # An EOF category avoids a fixed-size length header. It is only emitted
    # after the same exclusion backoff used for a byte, so the decoder can
    # stop without any out-of-band state.
    excluded = set()
    for order in range(min(_MAX_ORDER, len(history)), 0, -1):
        row = model.row_for(order, history)
        if row is None:
            continue
        choices = model._choices(row, excluded)
        if not choices:
            continue
        choices.append((_ESC, model._escape(order, row)))
        _encode_with_total(coder, choices, _ESC)
        for represented, frequency in row.items():
            if represented != _ESC and frequency:
                excluded.add(represented)
    choices = [(symbol0, frequency) for symbol0, frequency in
               enumerate(model.order0) if symbol0 not in excluded]
    _encode_with_total(coder, choices, _EOF)
    return coder.finish()


def decompress(blob: bytes) -> bytes:
    if not isinstance(blob, (bytes, bytearray)) or not blob:
        raise ValueError("truncated PPM stream")
    model = _PPM()
    decoder = _Decoder(blob)
    history = bytearray()
    output = bytearray()

    while True:
        excluded = set()
        symbol = None
        for order in range(min(_MAX_ORDER, len(history)), 0, -1):
            row = model.row_for(order, history)
            if row is None:
                continue
            choices = model._choices(row, excluded)
            if not choices:
                continue
            choices.append((_ESC, model._escape(order, row)))
            chosen = decoder.choose(choices, sum(frequency for _, frequency in choices))
            if chosen != _ESC:
                symbol = chosen
                break
            for represented, frequency in row.items():
                if represented != _ESC and frequency:
                    excluded.add(represented)

        if symbol is None:
            choices = [(symbol0, frequency) for symbol0, frequency in
                       enumerate(model.order0) if symbol0 not in excluded]
            symbol = decoder.choose(choices, sum(frequency for _, frequency in choices))
        if symbol == _EOF:
            break
        output.append(symbol)
        model.update(history, symbol)
        history.append(symbol)

    return bytes(output)
