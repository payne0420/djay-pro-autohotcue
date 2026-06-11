"""Parser and serializer for Algoriddim djay's TSAF binary serialization format.

TSAF is the object-archive format used for blobs in djay's YapDatabase-backed
MediaLibrary.db (table ``database2``). Reverse-engineered from observation;
validated by byte-perfect round-trips over every blob in a real library.

Format overview
---------------
Header (20 bytes):
    magic   "TSAF"
    u16 x2  version (3, 3)
    u64     container count (objects + arrays + sets in the stream)
    u32     interned-string count

Token stream follows. Multi-byte scalar payloads are aligned to their natural
boundary (4 or 8 bytes) relative to the file start with zero padding.

Tags:
    0x00  end-of-object marker
    0x02  integer, inline u8 payload (variant tag)
    0x05  string back-reference (u8 index into intern table)
    0x08  inline string: NUL-terminated UTF-8, appended to intern table
    0x0a  array (u32 count, 4-aligned) -- mutable variant
    0x0b  array (u32 count, 4-aligned) -- immutable variant
    0x0c  array (u32 count, 4-aligned) -- variant seen for queue itemUUIDs
    0x0d  bool true
    0x0e  bool false
    0x0f  integer, inline u8 payload
    0x10  integer, u16 payload, 2-aligned
    0x11  integer, u32 payload, 4-aligned
    0x13  float32, 4-aligned
    0x15  data: u32 length (4-aligned) + raw bytes + zero padding to a
          4-byte boundary
    0x19  ordered set (u32 count, 4-aligned)
    0x1a  set (u32 count, 4-aligned)
    0x21  URL: string token + 0x00 terminator (wrapper object)
    0x2b  object begin: followed by class-name string token, then
          (value, field-name) pairs, terminated by 0x00
    0x2d  integer 0, no payload
    0x2e  integer 1, no payload
    0x30  float64, 8-aligned (NSDate)

Field order within an object is value first, then field name. The name is
optional: if the token after a value is not a string token, the field is
anonymous. The header container count tallies objects, arrays, sets, NSDate
(f64) and NSData values; URLs, strings and primitive scalars do not count.
Verified exact across an entire real library (38,385 blobs).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


MAGIC = b"TSAF"

TAG_END = 0x00
TAG_INT8_V2 = 0x02
TAG_REF = 0x05
TAG_STR = 0x08
TAG_ARRAY_A = 0x0A
TAG_ARRAY_B = 0x0B
TAG_ARRAY_C = 0x0C
TAG_TRUE = 0x0D
TAG_FALSE = 0x0E
TAG_INT8 = 0x0F
TAG_INT16 = 0x10
TAG_INT32 = 0x11
TAG_F32 = 0x13
TAG_DATA = 0x15
TAG_OSET = 0x19
TAG_SET = 0x1A
TAG_URL = 0x21
TAG_OBJ = 0x2B
TAG_M2D = 0x2D
TAG_M2E = 0x2E
TAG_F64 = 0x30

CONTAINER_TAGS = (TAG_ARRAY_A, TAG_ARRAY_B, TAG_ARRAY_C, TAG_OSET, TAG_SET)


@dataclass
class Obj:
    classname: str
    fields: list[tuple[str, object]] = field(default_factory=list)

    def get(self, name, default=None):
        for k, v in self.fields:
            if k == name:
                return v
        return default

    def set(self, name, value):
        for i, (k, _) in enumerate(self.fields):
            if k == name:
                self.fields[i] = (name, value)
                return
        self.fields.append((name, value))

    def __repr__(self):
        inner = ", ".join(f"{k}={v!r}" for k, v in self.fields)
        return f"{self.classname}({inner})"


@dataclass
class Arr:
    tag: int
    items: list = field(default_factory=list)

    def __repr__(self):
        return f"Arr[{self.tag:#04x}]({self.items!r})"


@dataclass
class Url:
    value: str


@dataclass
class Data:
    value: bytes

    def __repr__(self):
        return f"Data({len(self.value)}B)"


@dataclass
class F32:
    raw: bytes  # 4 bytes, little-endian, kept raw for exact round-trips

    @classmethod
    def of(cls, value: float) -> "F32":
        return cls(struct.pack("<f", value))

    @property
    def value(self) -> float:
        return struct.unpack("<f", self.raw)[0]

    def __repr__(self):
        return f"F32({self.value})"


@dataclass
class F64:
    raw: bytes  # 8 bytes

    @classmethod
    def of(cls, value: float) -> "F64":
        return cls(struct.pack("<d", value))

    @property
    def value(self) -> float:
        return struct.unpack("<d", self.raw)[0]

    def __repr__(self):
        return f"F64({self.value})"


@dataclass
class Int:
    tag: int  # TAG_INT8, TAG_INT8_V2, TAG_INT16 or TAG_INT32
    value: int

    def __repr__(self):
        return f"Int[{self.tag:#04x}]({self.value})"


@dataclass
class Marker:
    """Payload-less scalar (tags 0x2d / 0x2e); semantics opaque, preserved."""

    tag: int

    def __repr__(self):
        return f"Marker[{self.tag:#04x}]"


@dataclass
class Document:
    version: tuple[int, int]
    root: object


class ParseError(ValueError):
    pass


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0
        self.strings: list[str] = []

    def align(self, n: int):
        while self.pos % n:
            if self.buf[self.pos] != 0:
                raise ParseError(f"nonzero pad byte at {self.pos}")
            self.pos += 1

    def u8(self) -> int:
        v = self.buf[self.pos]
        self.pos += 1
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.buf, self.pos)[0]
        self.pos += 2
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.buf, self.pos)[0]
        self.pos += 8
        return v

    def raw(self, n: int) -> bytes:
        v = self.buf[self.pos:self.pos + n]
        if len(v) != n:
            raise ParseError("unexpected EOF in raw read")
        self.pos += n
        return v

    def cstr(self) -> str:
        end = self.buf.index(b"\x00", self.pos)
        s = self.buf[self.pos:end].decode("utf-8")
        self.pos = end + 1
        return s

    def string_token(self) -> str:
        tag = self.u8()
        if tag == TAG_STR:
            s = self.cstr()
            self.strings.append(s)
            return s
        if tag == TAG_REF:
            idx = self.u8()
            if idx >= len(self.strings):
                raise ParseError(f"string backref {idx} out of range")
            return self.strings[idx]
        raise ParseError(f"expected string token, got tag {tag:#04x} at {self.pos - 1}")

    def value(self, tag: int):
        if tag == TAG_OBJ:
            obj = Obj(self.string_token())
            while True:
                vtag = self.u8()
                if vtag == TAG_END:
                    return obj
                val = self.value(vtag)
                name = None
                if self.pos < len(self.buf) and self.buf[self.pos] in (TAG_STR, TAG_REF):
                    name = self.string_token()
                obj.fields.append((name, val))
        if tag in CONTAINER_TAGS:
            self.align(4)
            n = self.u32()
            arr = Arr(tag)
            for _ in range(n):
                arr.items.append(self.value(self.u8()))
            return arr
        if tag == TAG_STR:
            s = self.cstr()
            self.strings.append(s)
            return s
        if tag == TAG_REF:
            idx = self.u8()
            if idx >= len(self.strings):
                raise ParseError(f"string backref {idx} out of range")
            return self.strings[idx]
        if tag == TAG_F32:
            self.align(4)
            return F32(self.raw(4))
        if tag == TAG_F64:
            self.align(8)
            return F64(self.raw(8))
        if tag in (TAG_INT8, TAG_INT8_V2):
            return Int(tag, self.u8())
        if tag == TAG_INT16:
            self.align(2)
            return Int(tag, self.u16())
        if tag == TAG_INT32:
            self.align(4)
            return Int(tag, self.u32())
        if tag == TAG_DATA:
            self.align(4)
            n = self.u32()
            d = Data(self.raw(n))
            self.align(4)
            return d
        if tag == TAG_URL:
            u = Url(self.string_token())
            if self.u8() != TAG_END:
                raise ParseError(f"URL missing terminator at {self.pos - 1}")
            return u
        if tag == TAG_TRUE:
            return True
        if tag == TAG_FALSE:
            return False
        if tag in (TAG_M2D, TAG_M2E):
            return Marker(tag)
        raise ParseError(f"unknown tag {tag:#04x} at {self.pos - 1}")


def parse(buf: bytes) -> Document:
    if buf[:4] != MAGIC:
        raise ParseError("bad magic")
    r = _Reader(buf)
    r.pos = 4
    v1, v2 = r.u16(), r.u16()
    ncontainers = r.u64()
    nstrings = r.u32()
    root = r.value(r.u8())
    if r.pos != len(buf):
        raise ParseError(f"trailing bytes at {r.pos}/{len(buf)}")
    if len(r.strings) != nstrings:
        raise ParseError(f"string count mismatch: header {nstrings}, got {len(r.strings)}")
    if _count_containers(root) != ncontainers:
        raise ParseError(
            f"container count mismatch: header {ncontainers}, got {_count_containers(root)}"
        )
    return Document((v1, v2), root)


def _count_containers(node) -> int:
    if isinstance(node, Obj):
        return 1 + sum(_count_containers(v) for _, v in node.fields)
    if isinstance(node, Arr):
        return 1 + sum(_count_containers(v) for v in node.items)
    if isinstance(node, (Data, F64)):
        return 1
    return 0


class _Writer:
    def __init__(self):
        self.out = bytearray()
        self.strings: dict[str, int] = {}

    def align(self, n: int):
        while len(self.out) % n:
            self.out.append(0)

    def u8(self, v: int):
        self.out.append(v)

    def u16(self, v: int):
        self.out += struct.pack("<H", v)

    def u32(self, v: int):
        self.out += struct.pack("<I", v)

    def string_token(self, s: str):
        idx = self.strings.get(s)
        if idx is None:
            self.strings[s] = len(self.strings)
            self.u8(TAG_STR)
            self.out += s.encode("utf-8") + b"\x00"
        else:
            if idx > 0xFF:
                raise ValueError(f"string backref index {idx} exceeds u8")
            self.u8(TAG_REF)
            self.u8(idx)

    def value(self, node):
        if isinstance(node, Obj):
            self.u8(TAG_OBJ)
            self.string_token(node.classname)
            for name, val in node.fields:
                self.value(val)
                if name is not None:
                    self.string_token(name)
            self.u8(TAG_END)
        elif isinstance(node, Arr):
            self.u8(node.tag)
            self.align(4)
            self.u32(len(node.items))
            for item in node.items:
                self.value(item)
        elif isinstance(node, str):
            self.string_token(node)
        elif isinstance(node, F32):
            self.u8(TAG_F32)
            self.align(4)
            self.out += node.raw
        elif isinstance(node, F64):
            self.u8(TAG_F64)
            self.align(8)
            self.out += node.raw
        elif isinstance(node, Int):
            self.u8(node.tag)
            if node.tag == TAG_INT32:
                self.align(4)
                self.u32(node.value)
            elif node.tag == TAG_INT16:
                self.align(2)
                self.u16(node.value)
            else:
                self.u8(node.value)
        elif isinstance(node, Data):
            self.u8(TAG_DATA)
            self.align(4)
            self.u32(len(node.value))
            self.out += node.value
            self.align(4)
        elif isinstance(node, Url):
            self.u8(TAG_URL)
            self.string_token(node.value)
            self.u8(TAG_END)
        elif node is True:
            self.u8(TAG_TRUE)
        elif node is False:
            self.u8(TAG_FALSE)
        elif isinstance(node, Marker):
            self.u8(node.tag)
        else:
            raise ValueError(f"cannot serialize {type(node)}")


def serialize(doc: Document) -> bytes:
    w = _Writer()
    # Header placeholder; counts patched after body is written.
    w.out += MAGIC
    w.out += struct.pack("<HH", *doc.version)
    w.out += struct.pack("<Q", _count_containers(doc.root))
    w.out += struct.pack("<I", 0)
    w.value(doc.root)
    struct.pack_into("<I", w.out, 16, len(w.strings))
    return bytes(w.out)
