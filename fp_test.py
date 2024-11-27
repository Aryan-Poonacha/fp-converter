import torch, unittest
from fp import round_to_fp8_represented_as_int8, undo_int8_fp8

class TestFPTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        print('Setting up test class.')
        # torch.manual_seed(1)

    def test_conversion_cycle_m2(self):
        original = 769
        t = torch.tensor([original], dtype=torch.bfloat16)
        u1 = round_to_fp8_represented_as_int8(t,2)
        print("FP Representation:", u1)
        u2 = undo_int8_fp8(u1,2)
        print("Reconverted original Representation:", u2)
        self.assertAlmostEqual(original, u2.item(), -2)

    def test_conversion_cycle_m3(self):
        original = 55
        t = torch.tensor([original], dtype=torch.bfloat16)
        u1 = round_to_fp8_represented_as_int8(t,3)
        print("FP Representation:", u1)
        u2 = undo_int8_fp8(u1,3)
        print("Reconverted original Representation:", u2)
        self.assertAlmostEqual(original, u2.item(), -2)

    def test_special_values(self):
        """Test handling of special values for both E4M3 and E5M2 formats"""
        for n_mantissa in [2, 3]:  # Test both E4M3 and E5M2
            # Test NaN
            x = torch.tensor([float('nan')], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.isnan(back))
            
            # Test infinities
            x = torch.tensor([float('inf')], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            if n_mantissa == 2:  # E5M2 supports infinity
                self.assertTrue(torch.isinf(back))
            else:  # E4M3 converts infinity to NaN
                self.assertTrue(torch.isnan(back))
            
            # Test zeros
            x = torch.tensor([0.0, -0.0], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.all(back == x))

    def test_subnormal_values(self):
        """Test handling of subnormal values for both formats"""
        for n_mantissa in [2, 3]:
            is_e4m3 = n_mantissa == 3
            # Test smallest subnormal
            min_subnormal = 2**(-9) if is_e4m3 else 2**(-16)
            x = torch.tensor([min_subnormal], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.abs(back - x) <= min_subnormal)
            
            # Test largest subnormal
            max_subnormal = 0.875 * 2**(-6) if is_e4m3 else 0.75 * 2**(-14)
            x = torch.tensor([max_subnormal], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.abs(back - x) <= max_subnormal * 0.1)

    def test_range_limits(self):
        """Test handling of values at the format limits"""
        for n_mantissa in [2, 3]:
            is_e4m3 = n_mantissa == 3
            # Test maximum normal value
            max_normal = 448.0 if is_e4m3 else 57344.0
            x = torch.tensor([max_normal], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.abs(back - x) <= max_normal * 0.1)
            
            # Test minimum normal value
            min_normal = 2**(-6) if is_e4m3 else 2**(-14)
            x = torch.tensor([min_normal], dtype=torch.bfloat16)
            fp8 = round_to_fp8_represented_as_int8(x, n_mantissa)
            back = undo_int8_fp8(fp8, n_mantissa)
            self.assertTrue(torch.abs(back - x) <= min_normal * 0.1)
        

    def test_expectation(self):
        val = 1 + 1/64
        n_trials = 50
        n_conversions = 256
        t = torch.tensor([val], dtype=torch.bfloat16)
        n_bits = 3

        all_sums = []

        for _ in range(n_trials):
            compressed_tensors = []
            for _ in range(n_conversions):
                u1 = round_to_fp8_represented_as_int8(t,n_bits)
                compressed_tensors.append(undo_int8_fp8(u1,n_bits))
            all_stack = torch.stack(compressed_tensors)
            all_sums.append(torch.sum(all_stack))
        
        std, mean = torch.std_mean(torch.stack(all_sums))

        
        """
        If we run 256 conversions of 1 + 1/64, we should expect 256 * 256/64 as the result (260).
        
        - 1/64 represents the 6th bit of a float32 mantissa or a 1/2**2 chance of adding a bit when converting to fp8
        - torch.rand_like() samples from a uniform distribution [0,1)
        - Repeating the process enough (50) times will see the mean of the results shift towards 260, even in spite of randomness.

        """ 
        self.assertAlmostEqual(mean.item(), val * n_conversions, -1)

    def test_one_overflow(self):
        t = torch.tensor([0.999999940395], dtype=torch.bfloat16)
        n_bits = 3

        u1 = round_to_fp8_represented_as_int8(t,n_bits)
        u2 = undo_int8_fp8(u1,n_bits)

        #The 1-bits should overflow into either of these numbers
        self.assertTrue(u2.item() == 1 or u2.item() == 0.96875)

    def test_twobit_mantissa(self):
        """Evaluate two bit mantissa. The only fractionals supported in two bits are 0, .25, .50, .75
        """
        t = torch.tensor([1.101], dtype=torch.bfloat16)

        mantissa_size = 2
        possible_values = [1.0, 1.25]
        for _ in range(10):
            u1 = round_to_fp8_represented_as_int8(t,mantissa_size)
            u2 = undo_int8_fp8(u1,mantissa_size)
            self.assertIn(u2.item(), possible_values)

    def test_threebit_mantissa(self):
        """Evaluate three bit mantissa
        """
        t = torch.tensor([1.101], dtype=torch.bfloat16)

        mantissa_size = 3
        possible_values = [1.0, 1.125]
        for _ in range(10):
            u1 = round_to_fp8_represented_as_int8(t,mantissa_size)
            u2 = undo_int8_fp8(u1,mantissa_size)
            self.assertIn(u2.item(), possible_values)
    
    def test_one_bit_exponent(self):
        """Evaluate a one bit exponent (by setting mantissa to 5)
        """
        t = torch.tensor([8.0], dtype=torch.bfloat16)

        mantissa_size = 5
        possible_values = [1,1.5]
        for _ in range(10):
            u1 = round_to_fp8_represented_as_int8(t,mantissa_size)
            u2 = undo_int8_fp8(u1,mantissa_size)
            
            # One bit exponent has a bias of -1
            # If the exponent bit is 0, the exponent is 2**-1 or 0.5
            # The 8 bit overflows into the sign position making the number negative
            self.assertEqual(u2.item(), -0.5)

    def test_rounding_rounding_fp8_uint8(self):
        src_dt = torch.bfloat16
        for n_mantissa in [3]:
            x = torch.full((10,), fill_value=0.5, dtype=src_dt)
            n_exp_bits = 7 - n_mantissa
            x = ((x - 0.5) * 2)
            x *= ((2 ** n_exp_bits) * (((2 ** n_mantissa) - 1) / (2 ** n_mantissa))) * 0.9
            output = round_to_fp8_represented_as_int8(x, n_mantissa)
        
if __name__ == '__main__':
    unittest.main()