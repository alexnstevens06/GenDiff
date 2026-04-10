import builtins

class ContinuousBitStream:
    """
    A bit stream backed by a bytearray that natively supports reading, writing, 
    and arbitrary rewinding. It uses big-endian bit packing within each byte.
    
    This abstracts away all byte boundaries, treating the entire payload as a 
    singular ribbon of bits.
    """
    def __init__(self, initial_bytes=b""):
        self._buffer = bytearray(initial_bytes)
        self._bit_pos = 0

    @property
    def bit_position(self) -> int:
        return self._bit_pos

    @bit_position.setter
    def bit_position(self, pos: int):
        if pos < 0:
            raise ValueError("Position cannot be negative")
        self._bit_pos = pos

    @property
    def num_bits_total(self) -> int:
        return len(self._buffer) * 8

    def write(self, bit: int):
        """Write a single bit (0 or 1)."""
        if bit not in (0, 1):
            raise ValueError("Argument must be 0 or 1")
            
        byte_idx = self._bit_pos // 8
        bit_idx = 7 - (self._bit_pos % 8)
        
        # Grow buffer if needed
        if byte_idx >= len(self._buffer):
            self._buffer.append(0)
            
        if bit:
            self._buffer[byte_idx] |= (1 << bit_idx)
        else:
            self._buffer[byte_idx] &= ~(1 << bit_idx)
            
        self._bit_pos += 1

    def write_bits(self, bits):
        """Write an iterable of bits."""
        for b in bits:
            self.write(b)

    def write_uint(self, val: int, num_bits: int):
        """Write an unsigned integer of a specific bit width."""
        for i in range(num_bits - 1, -1, -1):
            self.write((val >> i) & 1)

    def read(self) -> int:
        """Read a single bit. Returns -1 on EOF."""
        byte_idx = self._bit_pos // 8
        bit_idx = 7 - (self._bit_pos % 8)
        
        if byte_idx >= len(self._buffer):
            return -1
            
        bit = (self._buffer[byte_idx] >> bit_idx) & 1
        self._bit_pos += 1
        return bit

    def read_no_eof(self) -> int:
        """Read a single bit, raising EOFError on EOF."""
        b = self.read()
        if b == -1:
            raise EOFError("Unexpected end of ContinuousBitStream")
        return b

    def read_uint(self, num_bits: int) -> int:
        """Read an unsigned integer of a specific bit width."""
        val = 0
        for _ in range(num_bits):
            b = self.read()
            if b == -1:
                raise EOFError("Unexpected end of ContinuousBitStream during read_uint")
            val = (val << 1) | b
        return val

    def rewind(self, n: int):
        """Rewind the cursor by n bits."""
        if n < 0:
            raise ValueError("Cannot rewind by a negative amount")
        if self._bit_pos - n < 0:
            raise ValueError(f"Cannot rewind {n} bits, current position is {self._bit_pos}")
        self._bit_pos -= n

    def get_bytes(self) -> bytes:
        """Return the underlying bytearray as bytes."""
        return bytes(self._buffer)

    def close(self):
        """Compatibility signature."""
        pass
