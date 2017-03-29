# Class storing respective results
'''
iCENTER = 0
iLABELS = 1
iINIT_PARTITION = 2
iFIN_EUCLID = 3
iOBJ_HIST = 4
iITERS_EXEC = 5
iPART_COEFF = 6
iCOL_SCHEME = 7
'''
class ClusterResult:
    def __init__(self, results, path, name, extention):
        self.center = results[0]
        self.labels = results[1]
        self.init_partition = results[2]
        self.fin_euclid = results[3]
        self.obj_hist = results[4]
        self.iters_exec = results[5]
        self.part_coeef = results[6]
        self.col_scheme = results[7]
        self.src_path = path
        self.src_name = name
        self.src_extention = extention
