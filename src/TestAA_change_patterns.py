import unittest
import AA_change_patterns as patterns
class TestAA_change_patterns(unittest.TestCase):
    def test_frequent_range_delete(self):
        validPattern = "Î”141-144" 
        invalidPattern01 =  "Î”141-" 
        invalidPattern02 =  "Î”141-144A" 
        invalidPattern03 =  "”141-144" 
        output = patterns.frequent_range_delete(validPattern)
        invalid_output01 = patterns.frequent_range_delete(invalidPattern01)
        invalid_output02 = patterns.frequent_range_delete(invalidPattern02)
        invalid_output03 = patterns.frequent_range_delete(invalidPattern03)

        self.assertTrue(output)
        self.assertFalse(invalid_output01)
        self.assertFalse(invalid_output02)
        self.assertFalse(invalid_output03)

    def test_frequent_residue_delete(self):
        validPattern = "Î”146" 
        invalidPattern01 =  "Î”141-" 
        invalidPattern02 =  "Î”A-144A" 
        invalidPattern03 =  "”141-144" 
        output = patterns.freqeuent_single_residue_delete(validPattern)
        invalid_output01 = patterns.freqeuent_single_residue_delete(invalidPattern01)
        invalid_output02 = patterns.freqeuent_single_residue_delete(invalidPattern02)
        invalid_output03 = patterns.freqeuent_single_residue_delete(invalidPattern03)

        self.assertTrue(output)
        self.assertFalse(invalid_output01)
        self.assertFalse(invalid_output02)
        self.assertFalse(invalid_output03)
    def test_single_residue_subsitution(self):
        validPattern = "A555L" 
        invalidPattern01 = "AA55L"
        invalidPattern02 = "A55"
        invalidPattern03 = "55"
        output = patterns.single_residue_subsitution(validPattern)
        self.assertTrue(output)
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern01))
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern02))
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern03))

    def test_multiple_residue_subsitution(self):
        validPattern = "ACT55-7TPS" 
        invalidPattern01 = "AA55L"
        invalidPattern02 = "A55"
        invalidPattern03 = "55"
        output = patterns.multiple_residue_subsitution(validPattern)
        self.assertTrue(output)
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern01))
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern02))
        self.assertFalse(patterns.single_residue_subsitution(invalidPattern03))

    def test_delPattern_at_end(self):
        validPattern01 = "675-3677del" 
        validPattern02 = "f141del" 
        invalidPattern01 = "del555"
        self.assertTrue(patterns.delFound_at_end(validPattern01))
        self.assertTrue(patterns.delFound_at_end(validPattern02))
        self.assertFalse(patterns.delFound_at_end(invalidPattern01))

    def test_is_hypen_found(self):
        validPattern01 = "675-3677del" 
        invalidPattern01 = "del555"
        self.assertTrue(patterns.is_hypen_found(validPattern01))
        self.assertFalse(patterns.is_hypen_found(invalidPattern01))
    
    def test_is_alphabet_follows_digits(self):
        validPattern01 = "A3677" 
        validPattern02 = "A3"
        invalidPattern01 = "3677A"
        self.assertTrue(patterns.is_alphabet_follows_digits(validPattern01))
        self.assertTrue(patterns.is_alphabet_follows_digits(validPattern02))
        self.assertFalse(patterns.is_alphabet_follows_digits(invalidPattern01))

    def test_is_del_in_beginning(self):
        validPattern01 = "del241/243" 
        validPattern02 = "del241"
        invalidPattern01 = "241del"
        self.assertTrue(patterns.is_del_in_beginning(validPattern01))
        self.assertTrue(patterns.is_del_in_beginning(validPattern02))
        self.assertFalse(patterns.is_del_in_beginning(invalidPattern01))

    def test_is_nums_slash_nums(self):
        validPattern01 = "525/364" 
        validPattern02 = "5/4"
        invalidPattern01 = "del241-243"
        self.assertTrue(patterns.is_nums_slash_nums(validPattern01))
        self.assertTrue(patterns.is_nums_slash_nums(validPattern02))
        self.assertFalse(patterns.is_nums_slash_nums(invalidPattern01))

    def test_insert_in_between(self):
        validPattern01 =  "248aKTRNKSTSRRE248k"
        invalidPattern01 =  "248KTRNKSTSRRE248k"
        invalidPattern02 =  "248KTRNKSTSRRE248"
        self.assertTrue(patterns.insert_in_between(validPattern01))
        self.assertFalse(patterns.insert_in_between(invalidPattern01))
        self.assertFalse(patterns.insert_in_between(invalidPattern02))


if __name__=='__main__':
    unittest.main()