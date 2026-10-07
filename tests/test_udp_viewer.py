import struct

from udp_viewer import GDU_HEADER, GduFrameAssembler


def packet(frame_id: int, chunk_count: int, chunk_id: int, payload: bytes) -> bytes:
    return GDU_HEADER.pack(b"GDU1", frame_id, chunk_count, chunk_id) + payload


def test_reassembles_out_of_order_chunks() -> None:
    assembler = GduFrameAssembler()
    addr = ("192.168.1.2", 50000)
    assert assembler.add(packet(7, 3, 2, b"cc"), addr) is None
    assert assembler.add(packet(7, 3, 0, b"aa"), addr) is None
    assert assembler.add(packet(7, 3, 1, b"bb"), addr) == b"aabbcc"


def test_passes_through_unframed_jpeg() -> None:
    assembler = GduFrameAssembler()
    jpeg = b"\xff\xd8example\xff\xd9"
    assert assembler.add(jpeg, ("127.0.0.1", 1234)) == jpeg


def test_rejects_invalid_chunk_id() -> None:
    assembler = GduFrameAssembler()
    raw = struct.pack("!4sIHH", b"GDU1", 1, 2, 2) + b"bad"
    assert assembler.add(raw, ("127.0.0.1", 1234)) is None
