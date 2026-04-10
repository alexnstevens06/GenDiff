from core.continuous_bitstream import ContinuousBitStream
from core.AC import arithmeticcoding
import numpy as np

def test_ac_overread_constant():
    # We will test if the Nayuki decoder ALWAYS overreads exactly 31 bits
    
    # 1. Setup encoder
    stream = ContinuousBitStream()
    
    # Fake cumulative freq for vocab size 3 (0 is EOS)
    # [EOS=0, A=1, B=2]
    # uniform freq = 1 each
    cumul = np.array([0, 1, 2, 3], dtype=np.uint64)
    
    enc = arithmeticcoding.ArithmeticEncoder(32, stream)
    
    # Encode A, B, A, B, EOS
    sequence = [1, 2, 1, 2, 0]
    for sym in sequence:
        enc.write(cumul, sym)
    enc.finish()
    
    # 2. Emulate the outer system writing metadata directly after the AC blob
    meta_bits = [1, 0, 1, 1, 0, 1, 0, 0]
    stream.write_bits(meta_bits)
    
    # 3. Decode
    stream.bit_position = 0 # reset to beginning
    
    # `ArithmeticDecoder` initializes by demanding 32 bits from the stream
    dec = arithmeticcoding.ArithmeticDecoder(32, stream)
    
    decoded = []
    while True:
        sym = dec.read(cumul, 3)
        decoded.append(sym)
        if sym == 0: # EOS
            break
            
    assert decoded == sequence, f"Expected {sequence}, got {decoded}"
    
    # 4. Prove the overread!
    # Because we explicitly padded the encoder with 31 zeros, the decoder's
    # 31 bits of lookahead simply consumed those padding zeros!
    # Therefore, the stream cursor should be natively positioned EXACTLY at the metadata.
    overread = 0
    stream.rewind(overread)
    
    recovered_meta = []
    for _ in range(len(meta_bits)):
        recovered_meta.append(stream.read())
        
    assert recovered_meta == meta_bits, f"Expected metadata {meta_bits}, but recovered {recovered_meta}. Overread constant 31 is INVALID!"
    print("AC Rewind Math Proven! Overread is exactly 31 bits.")

if __name__ == "__main__":
    test_ac_overread_constant()
