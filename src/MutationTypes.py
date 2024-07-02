import enum
class MutationTypes(enum.Enum):
    NOT_RECOGNIZED = 0
    #Deletion
    DEL_SINGLE_RES = 1  # F140del
    DEL_RES_1_OR_2  = 2 #del241/243
    DEL_IN_RANGE  = 3 #242-244del
    DEL_SINGLE_RES_FREQ = 4 #Î”146
    DEL_IN_RANGE_FREQ  = 5 #Î”141-144

    #Insertion
    INSERT_IN_BETWEEN = 6 #248aKTRNKSTSRRE248k

    #Subsitution 
    SINGLE_RES_SUB = 7 #E484K
    MULTIPLE_RES_SUB = 8 #KVG444-6TST

    #Greate SIG
    GREANY_SIG = 9

if __name__ == "__main__":
    #loadAdditionalEscapes()
    print(MutationTypes.DEL_SINGLE_RES.value)