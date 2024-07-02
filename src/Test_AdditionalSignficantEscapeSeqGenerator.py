import unittest
from AA_change_patterns import multiple_residue_subsitution
from escape import read_wildSequence
import SignficantEscapeGenerator as SEG
from SignficantEscapeGenerator import read_sequences
from escape import ADDITIONAL_MUTANT_GENERATED_PATH
from MutationTypes import MutationTypes
import AdditionalSignficantEscapeSeqGenerator as ASG

class Test_AdditionalSignficantEscapeSeqGenerator(unittest.TestCase):
    def setUp(self):
        self.wild_seq = read_wildSequence()
        self.multiple_residue_subsitution_path = ADDITIONAL_MUTANT_GENERATED_PATH + \
            "/"+MutationTypes.MULTIPLE_RES_SUB.name+".faa"
        self.multiple_res_seqeunces = read_sequences(self.multiple_residue_subsitution_path)
    
    '''
    Length of generated subsitution file and original wild sequence must be equal
    '''
    def test_multiple_residue_subsitutionLengthEquality(self):
       for seq in self.multiple_res_seqeunces:
           assert len(self.wild_seq) == len(seq)


    def test_multiple_res_sub_residueReplaced_correctly(self):
        #KVG444-6TST
        subsitituted_seq_01 = self.multiple_res_seqeunces[0]
        print(len(subsitituted_seq_01))
        original_residues = "KVG"
        changed_residues= "TST"
        start_position = 443
        print(self.wild_seq[start_position])
        print(subsitituted_seq_01[start_position])
        #Verify K replace by T
        assert self.wild_seq[start_position] == 'K'
        assert subsitituted_seq_01[start_position] == 'T'

        #Verify V replaced by S
        assert self.wild_seq[start_position+1] == 'V'
        assert subsitituted_seq_01[start_position+1] == 'S'

        #Verify G replaced by T
        assert self.wild_seq[start_position+2] == 'G'
        assert subsitituted_seq_01[start_position+2] == 'T'

    '''
    When single residue is deleted from the sequence,
    -sequence is left shifed by 1 position
    so, the wild_sequence value at postion+1, should be equal to mutated sequence residue at position 
    And length of mutated sequence should be less by 1
    '''
    def test_single_res_delete(self):
        single_residue_del_path = ADDITIONAL_MUTANT_GENERATED_PATH + \
            "/"+MutationTypes.DEL_SINGLE_RES.name+".faa"
        seqeunces = read_sequences(single_residue_del_path)
        #F140 Del 
        assert self.wild_seq[140] == seqeunces[0][139]
        assert len(seqeunces[0]) == len(self.wild_seq)-1

    def test_is_single_residue_significant_seq(self):
        significant_seqs = SEG.read_sequences(SEG.SINGLE_RES_SIGNIFICANT_SEQS_PATH)
        result01 = ASG.is_single_residue_significant_seq(significant_seqs[0])
        assert(result01 == True)
      






if __name__ == '__main__':
    unittest.main()
