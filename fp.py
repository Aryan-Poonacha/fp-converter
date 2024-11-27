from typing import Optional
import torch

#@torch.jit.script
def round_to_fp8_represented_as_int8(
    t: torch.Tensor,
    n_mantissa: int,
    out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

    fp8_mantissa_bits = n_mantissa
    fp8_exponent_bits = 7 - n_mantissa
    
    bfloat16_mantissa_bits = 7
    bfloat16_exponent_bits = 8
    bfloat16_total_bits = 16
    
    #get the sign out
    sign = (t.view(torch.int16) >> (bfloat16_total_bits - 1)).to(torch.uint8)
    sign = torch.bitwise_and(sign, torch.ones_like(sign, dtype=torch.uint8))

    # sign bit abs()
    t_bit = torch.abs(t)

    base = (t_bit.view(torch.int16) >> bfloat16_mantissa_bits)
    mantissa = torch.bitwise_xor((base << bfloat16_mantissa_bits), t_bit.view(torch.int16))
    
    # calculate E()
    x = ((mantissa << fp8_mantissa_bits) % (2**bfloat16_mantissa_bits))
    y = torch.ones_like(t).view(torch.int16)
    z = torch.bitwise_or(x, y)
    exp = (z.view(t.dtype) - 1)

    #Stochastic rounding step
    rand = torch.rand_like(exp, dtype=torch.float32)
    ones = torch.ceil(exp.to(torch.float32) - rand).type(torch.uint8)

    # Combine components
    fp8_sign = sign << 7
    fp8_base = base - (2**(bfloat16_exponent_bits - 1) - 1) + (2**(fp8_exponent_bits - 1) - 1)
    fp8_base = (fp8_base << fp8_mantissa_bits).to(torch.uint8)
    fp8_mantissa = (mantissa >> (bfloat16_mantissa_bits - fp8_mantissa_bits)).to(torch.uint8)
    fp8_as_uint8 = fp8_sign + fp8_base + fp8_mantissa
    
    return fp8_as_uint8 + ones

#@torch.jit.script
def undo_int8_fp8(
    t: torch.ByteTensor,
    n_mantissa: int
) -> torch.Tensor:
    
    bfloat16_mantissa_bits = n_mantissa
    bfloat16_exponent_bits = 7 - n_mantissa
    
    #get sign out
    sign = t >> 7
    shifted_sign = (sign.type(torch.int16) << 15)
    
    # Extract and adjust base
    base_mantissa = t << 1
    base = (base_mantissa >> (bfloat16_mantissa_bits + 1)) - (2**(bfloat16_exponent_bits - 1) - 1)
    base = base.type(torch.int16) + (2**(5 - 1) - 1)
    shifted_base = base << 10

    # Extract mantissa
    mantissa = base_mantissa << bfloat16_exponent_bits
    shifted_mantissa = mantissa.type(torch.int16) << 2
    
    return (shifted_base + shifted_sign + shifted_mantissa).view(torch.float16)