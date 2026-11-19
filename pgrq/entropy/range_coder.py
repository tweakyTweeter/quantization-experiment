"""
Stage 9: Entropy coding using arithmetic coding.

Implements arithmetic coding for:
- Base codebook indices
- Residual codebook indices (per level)
- Phase indices
- Mode/class tags

Reports MEASURED compressed bytes, not theoretical entropy.
"""

import torch
from typing import List, Tuple, Dict, Optional
from collections import Counter
import struct
import math


class SimpleArithmeticCoder:
    """
    Simplified arithmetic coder for practical entropy coding.
    
    Uses a fixed-precision approach suitable for our use case.
    """
    
    PRECISION = 32
    FULL = 1 << PRECISION
    HALF = 1 << (PRECISION - 1)
    QUARTER = 1 << (PRECISION - 2)
    
    def __init__(self):
        self.low = 0
        self.high = self.FULL
        self.buffer = []
        self.pending_bits = 0
    
    def _output_bit(self, bit: int):
        """Output a bit and any pending bits."""
        self.buffer.append(bit)
        while self.pending_bits > 0:
            self.buffer.append(1 - bit)
            self.pending_bits -= 1
    
    def encode(self, symbols: List[int], cumfreqs: List[int], total: int) -> bytes:
        """
        Encode symbols using arithmetic coding.
        
        Args:
            symbols: List of symbol indices.
            cumfreqs: Cumulative frequency table.
            total: Total frequency.
            
        Returns:
            data: Compressed bytes.
        """
        self.low = 0
        self.high = self.FULL
        self.buffer = []
        self.pending_bits = 0
        
        for sym in symbols:
            range_size = self.high - self.low
            self.high = self.low + (range_size * cumfreqs[sym + 1]) // total
            self.low = self.low + (range_size * cumfreqs[sym]) // total
            
            while True:
                if self.high < self.HALF:
                    self._output_bit(0)
                    self.low = self.low * 2
                    self.high = self.high * 2
                elif self.low >= self.HALF:
                    self._output_bit(1)
                    self.low = (self.low - self.HALF) * 2
                    self.high = (self.high - self.HALF) * 2
                elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                    self.pending_bits += 1
                    self.low = (self.low - self.QUARTER) * 2
                    self.high = (self.high - self.QUARTER) * 2
                else:
                    break
        
        # Flush
        self.pending_bits += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)
        
        # Convert bits to bytes
        while len(self.buffer) % 8 != 0:
            self.buffer.append(0)
        
        data = []
        for i in range(0, len(self.buffer), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | self.buffer[i + j]
            data.append(byte)
        
        return bytes(data)
    
    def decode(self, data: bytes, cumfreqs: List[int], total: int, num_symbols: int) -> List[int]:
        """
        Decode symbols from compressed data.
        
        Args:
            data: Compressed bytes.
            cumfreqs: Cumulative frequency table.
            total: Total frequency.
            num_symbols: Number of symbols to decode.
            
        Returns:
            symbols: Decoded symbol list.
        """
        # Convert bytes to bits
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        
        bit_pos = 0
        
        def read_bit():
            nonlocal bit_pos
            if bit_pos < len(bits):
                b = bits[bit_pos]
                bit_pos += 1
                return b
            return 0
        
        # Initialize
        self.low = 0
        self.high = self.FULL
        code = 0
        for _ in range(self.PRECISION):
            code = (code << 1) | read_bit()
        
        symbols = []
        num_levels = len(cumfreqs) - 1
        
        for _ in range(num_symbols):
            range_size = self.high - self.low
            scaled_value = ((code - self.low + 1) * total - 1) // range_size
            
            # Binary search for symbol
            sym = 0
            for i in range(num_levels):
                if cumfreqs[i + 1] > scaled_value:
                    sym = i
                    break
                sym = i
            
            symbols.append(sym)
            
            self.high = self.low + (range_size * cumfreqs[sym + 1]) // total
            self.low = self.low + (range_size * cumfreqs[sym]) // total
            
            while True:
                if self.high < self.HALF:
                    self.low = self.low * 2
                    self.high = self.high * 2
                    code = (code * 2) | read_bit()
                elif self.low >= self.HALF:
                    self.low = (self.low - self.HALF) * 2
                    self.high = (self.high - self.HALF) * 2
                    code = ((code - self.HALF) * 2) | read_bit()
                elif self.low >= self.QUARTER and self.high < 3 * self.QUARTER:
                    self.low = (self.low - self.QUARTER) * 2
                    self.high = (self.high - self.QUARTER) * 2
                    code = ((code - self.QUARTER) * 2) | read_bit()
                else:
                    break
        
        return symbols


def build_frequency_table(
    symbols: torch.Tensor,
    num_symbols: int,
    min_freq: int = 1,
) -> Tuple[List[int], List[int], int]:
    """
    Build frequency and cumulative frequency tables from symbols.
    
    Args:
        symbols: Symbol tensor (will be flattened).
        num_symbols: Total number of possible symbols.
        min_freq: Minimum frequency per symbol (for smoothing).
        
    Returns:
        freqs: Frequency of each symbol.
        cumfreqs: Cumulative frequencies (length num_symbols + 1).
        total: Total frequency count.
    """
    symbols_flat = symbols.flatten().cpu().tolist()
    
    # Count frequencies
    counts = Counter(symbols_flat)
    
    # Build frequency table with smoothing
    freqs = [max(counts.get(i, 0), min_freq) for i in range(num_symbols)]
    
    # Build cumulative frequencies
    cumfreqs = [0]
    for f in freqs:
        cumfreqs.append(cumfreqs[-1] + f)
    
    total = cumfreqs[-1]
    
    return freqs, cumfreqs, total


def encode_symbols(
    symbols: torch.Tensor,
    num_symbols: int,
) -> Tuple[bytes, List[int]]:
    """
    Encode symbols using arithmetic coding.
    
    Args:
        symbols: Symbol tensor to encode.
        num_symbols: Total number of possible symbols.
        
    Returns:
        data: Compressed bytes.
        freqs: Frequency table (needed for decoding).
    """
    freqs, cumfreqs, total = build_frequency_table(symbols, num_symbols)
    
    coder = SimpleArithmeticCoder()
    symbol_list = symbols.flatten().cpu().tolist()
    symbol_list = [int(s) for s in symbol_list]
    
    data = coder.encode(symbol_list, cumfreqs, total)
    
    return data, freqs


def decode_symbols(
    data: bytes,
    freqs: List[int],
    num_symbols: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Decode symbols using arithmetic coding.
    
    Args:
        data: Compressed bytes.
        freqs: Frequency table from encoding.
        num_symbols: Number of symbols to decode.
        device: Target device for output tensor.
        
    Returns:
        symbols: Decoded symbol tensor.
    """
    # Build cumulative frequencies
    cumfreqs = [0]
    for f in freqs:
        cumfreqs.append(cumfreqs[-1] + f)
    total = cumfreqs[-1]
    
    coder = SimpleArithmeticCoder()
    symbols = coder.decode(data, cumfreqs, total, num_symbols)
    
    return torch.tensor(symbols, dtype=torch.long, device=device)


def compute_ideal_bits(symbols: torch.Tensor, num_symbols: int) -> float:
    """
    Compute ideal (entropy-based) bits for comparison.
    
    Args:
        symbols: Symbol tensor.
        num_symbols: Total number of possible symbols.
        
    Returns:
        ideal_bits: Theoretical minimum bits = -sum(p * log2(p)).
    """
    freqs, _, _ = build_frequency_table(symbols, num_symbols, min_freq=0)
    total = sum(freqs)
    
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for f in freqs:
        if f > 0:
            p = f / total
            entropy -= p * math.log2(p)
    
    return entropy * total


class SymbolStream:
    """
    Stream of multiple symbol types for entropy coding.
    
    Handles different symbol types (base codes, residual codes, etc.)
    with separate frequency tables.
    """
    
    def __init__(self):
        self.streams: Dict[str, Tuple[torch.Tensor, int]] = {}
    
    def add(self, name: str, symbols: torch.Tensor, num_symbols: int):
        """Add a symbol stream."""
        self.streams[name] = (symbols, num_symbols)
    
    def encode_all(self) -> Tuple[bytes, Dict]:
        """
        Encode all streams.
        
        Returns:
            data: Combined compressed bytes.
            metadata: Dict with per-stream info for decoding.
        """
        metadata = {
            "streams": {},
            "offsets": {},
        }
        
        all_data = []
        offset = 0
        
        for name, (symbols, num_symbols) in self.streams.items():
            stream_data, freqs = encode_symbols(symbols, num_symbols)
            
            metadata["streams"][name] = {
                "freqs": freqs,
                "num_symbols": num_symbols,
                "count": symbols.numel(),
                "shape": list(symbols.shape),
            }
            metadata["offsets"][name] = (offset, len(stream_data))
            
            all_data.append(stream_data)
            offset += len(stream_data)
        
        return b"".join(all_data), metadata
    
    @staticmethod
    def decode_all(data: bytes, metadata: Dict, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Decode all streams.
        
        Args:
            data: Combined compressed bytes.
            metadata: Metadata from encoding.
            device: Target device.
            
        Returns:
            streams: Dict mapping name -> decoded tensor.
        """
        streams = {}
        
        for name, info in metadata["streams"].items():
            start, length = metadata["offsets"][name]
            stream_data = data[start:start + length]
            
            symbols = decode_symbols(
                stream_data,
                info["freqs"],
                info["count"],
                device=device,
            )
            
            streams[name] = symbols.view(info["shape"])
        
        return streams
