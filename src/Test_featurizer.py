import unittest
from escape import read_wildSequence
import featurizer as featurizer
class Test_featurizer(unittest.TestCase):
    def test_sequence_int_encoding(self):
        seq_list = []
        wild_seq = read_wildSequence()
        seq_list.append(wild_seq)
        vocab = featurizer.getVocabDictionary()
        seqs = featurizer.sequence_int_encoding(seq_list)

        seq = seqs[0]
        assert(len(seq_list[0]) == len(seq))

        firstIndex_from_wild_seq = vocab[wild_seq[0]]
        firstIndex_from_returned_seq = seq[0]

        
        lastIndex_from_wild_seq = vocab[wild_seq[len(wild_seq)-1 ]]
        lastValue_from_returned_seq = seq.pop()

        assert(firstIndex_from_wild_seq == firstIndex_from_returned_seq )
        assert(lastIndex_from_wild_seq == lastValue_from_returned_seq )






if __name__=='__main__':
    unittest.main()