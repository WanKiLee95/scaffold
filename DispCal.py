import numpy as np
import time
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import psutil
import gc
import sys
import pandas as pd

def DisplacementCalculation(NodeNumber, Support, F_GlobalCordi, K_Global, NodeGenCordi, ScaleFactor, pinNodeList, K_Size):

    BoundaryCond = np.ones(K_Size)
    for i in range(NodeNumber):
        for j in range(6):
            if Support[i] == 1: # 서포트 유
                BoundaryCond[6*i+j] = 0
            else: #서포트 무
                BoundaryCond[6*i+j] = 1
    for i in range(int((K_Size-6*NodeNumber)/3)):
        BoundaryCond[6*NodeNumber+3*i]=0

    NumberOfBC = np.count_nonzero(BoundaryCond==0)
    uLocal_GlobalCordi = np.zeros((K_Size-NumberOfBC,1))
    u_DeformedGlobal = np.zeros((K_Size,1))
    Node_Deformed = np.zeros((NodeNumber,3))
    BC_index = np.nonzero(BoundaryCond==0)[0]
    nonBC_index = np.nonzero(BoundaryCond==1)[0]
    j=0
    FLocal_GlobalCordi = F_GlobalCordi #F_GlobalCordi[nonBC_index]

    K_Global = csc_matrix(K_Global)
    K_Global = K_Global[:,nonBC_index]
    K_Global = csr_matrix(K_Global)
    K_Global = K_Global[nonBC_index,:]

    K_AppliedBC = K_Global.toarray()
    torsionIdx = np.argwhere(np.all(K_AppliedBC[:]==0,axis=0))
    K_AppliedBC = np.delete(K_AppliedBC, torsionIdx, axis=0)
    K_AppliedBC = np.delete(K_AppliedBC, torsionIdx, axis=1)
    # FLocal_GlobalCordi = np.delete(FLocal_GlobalCordi, torsionIdx, axis=0)
    uLocal_GlobalCordi = np.delete(uLocal_GlobalCordi, torsionIdx, axis=0)

    detValue = np.linalg.det(K_AppliedBC)
    print('K_Global determinant : ', detValue)

    K_Global = csr_matrix(K_AppliedBC)

    uLocal_GlobalCordi = spsolve(K_Global, FLocal_GlobalCordi)
    u_AppliedBC = uLocal_GlobalCordi

    for i in range(torsionIdx.shape[0]):
        uLocal_GlobalCordi = np.insert(uLocal_GlobalCordi, torsionIdx[i,0],0, axis=0)

    for idx, val in enumerate(nonBC_index):
        u_DeformedGlobal[val] = uLocal_GlobalCordi[idx]

    for i in range(NodeNumber):
        for j in range(3):
            Node_Deformed[i,j] = NodeGenCordi[i,j] + ScaleFactor*u_DeformedGlobal[6*i+j]
    
    return K_AppliedBC, Node_Deformed, u_DeformedGlobal, u_AppliedBC
