from core.continuous_bitstream import ContinuousBitStream

def test_continuous_bitstream():
    stream = ContinuousBitStream()

    # 1. Write basic bits
    stream.write(1)
    stream.write(0)
    stream.write(1)
    stream.write(1)
    assert stream.bit_position == 4
    
    # 2. Write uint wrapping across a byte boundary
    # We are at bit 4. Let's write an 8-bit integer: 0b10101010 (170)
    # This will cross from byte 0 to byte 1.
    stream.write_uint(170, 8)
    assert stream.bit_position == 12
    
    # Check underlying bytes
    b = stream.get_bytes()
    assert len(b) == 2
    # Bits should be: 1 0 1 1 (first 4) | 1 0 1 0 (from 170) -> 0b10111010 = 186
    # Byte 1 bits: 1 0 1 0 (rest of 170) | 0 0 0 0 -> 0b10100000 = 160
    assert b[0] == 186
    assert b[1] == 160
    
    # 3. Read back seamlessly
    stream.bit_position = 0
    assert stream.read() == 1
    assert stream.read() == 0
    assert stream.read() == 1
    assert stream.read() == 1
    
    uint_read = stream.read_uint(8)
    assert uint_read == 170
    assert stream.bit_position == 12
    
    # 4. End of stream
    # Since we allocated 2 bytes (16 bits), the remaining 4 bits are 0 padding.
    assert stream.read() == 0
    assert stream.read() == 0
    assert stream.read() == 0
    assert stream.read() == 0
    # Now we are at bit 16, which exceeds the 2 bytes, so we get -1
    assert stream.read() == -1
    
    # 5. Overwrite and Rewind test
    stream.rewind(4) # rewinds from 16 to 12
    assert stream.bit_position == 12
    # Let's rewind back to 8 to test the overwrite
    stream.rewind(4) # now at 8
    assert stream.bit_position == 8
    # We should have unread exactly the last 4 bits of the 170 (which were 1010)
    # Let's read them to prove it
    assert stream.read() == 1
    assert stream.read() == 0
    assert stream.read() == 1
    assert stream.read() == 0
    assert stream.bit_position == 12
    
    print("All ContinuousBitStream tests passed successfully!")

if __name__ == "__main__":
    test_continuous_bitstream()
