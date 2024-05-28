import numpy as np
import KMatrix
import DispCal
import pandas as pd

ScaleFactor = 1
NodeNumber=7
Elasticity = np.ones((7,1)) * 2E8 # kPa
I_xyz = np.ones((7,3)) * 0.0001977 # m4
Area = np.ones((7,1)) * 0.01097 # m2
G_Shear = np.ones((7,1))*76923076.92307693 # kPa
K_Size = 66 # Node개수(7)*6 + pin joint 연결 수(8)*3
NodeGenCordi = np.array([[0,0,0],[0,0,10],[6,0,10],[12,0,0],[12,0,10],[0,0,20],[12,0,20]]) #노드 좌표
ElementData = np.array([[0,1,10],[1,2,6],[2,4,6],[3,4,10],[1,5,10],[4,6,10],[5,6,12]]) #노드1번호, 노드2번호, 길이
Support = np.array([1,0,0,1,0,0,0]).T

F_GlobalCordi = np.array([0,0,-112.5,0,0,-225,0,0,-112.5,0,0,-112.5,0,225,0,0,0,-112.5,0,-225,0,0,0,112.5,0,-112.5,0,112.5,0,-112.5,0,0,0,0,0,0,0]).T
# 경계조건 다 입힌 최종 하중 행렬

pinNodeList = [1, 2, 4]

K_Global = KMatrix.Gloabl_K(NodeNumber, ElementData, NodeGenCordi, Elasticity, G_Shear, I_xyz, Area, pinNodeList, K_Size)
K_AppliedBC, Node_Deformed, u_DeformedGlobal, u_AppliedBC = DispCal.DisplacementCalculation(NodeNumber, Support, F_GlobalCordi, K_Global, NodeGenCordi, ScaleFactor, pinNodeList, K_Size)


# 출력
K_Global = pd.DataFrame(K_Global)
K_Global.to_csv('K_Global.txt')

K_AppliedBC = pd.DataFrame(K_AppliedBC)
K_AppliedBC.to_csv('K_AppliedBC.txt')

F_GlobalCordi = pd.DataFrame(F_GlobalCordi)
F_GlobalCordi.to_csv('F_AppliedBC.txt')

u_AppliedBC = pd.DataFrame(u_AppliedBC)
u_AppliedBC.to_csv('u_AppliedBC.txt')
