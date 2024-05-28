import numpy as np

def Gloabl_K(NodeNumber, ElementData, NodeGenCordi, Elasticity, G_Shear, I_xyz, Area, pinNodeList, K_Size):
    K_Global_GlobalCordi=np.zeros((K_Size, K_Size))
    PartialRoatation = np.eye(K_Size)
    lmprint=np.zeros((1,5))
    pinCount = 0

    section_j = np.zeros((12,12))
    section_k = np.zeros((12,12))
    section_jk = np.zeros((12,12))
    section_kj = np.zeros((12,12))

    for i in range(len(ElementData)):
        j = int(ElementData[i,0])
        k = int(ElementData[i,1])
        Length = ElementData[i,2]

        K_Local = Local_K(j, k, Length, Elasticity, G_Shear, I_xyz, Area, pinNodeList)
        Rotation = RotationMat(j, k, Length, NodeGenCordi)

        j_pinFlag, k_pinFlag, l, m, pinCount = pinIndexing(j, k, NodeNumber, pinNodeList, pinCount)

        section_j[3:6,:] = K_Local[3:6,:]
        section_j[:,3:6] = K_Local[:,3:6]
        section_k[9:12,:] = K_Local[9:12,:]
        section_k[:,9:12] = K_Local[:,9:12]
        section_jk[3:6,9:12] = K_Local[3:6,9:12]
        section_kj[9:12,3:6] = K_Local[9:12,3:6]

        K_Local1 = K_Local - section_j*j_pinFlag - section_k*k_pinFlag + (section_jk+section_kj)*j_pinFlag*k_pinFlag
        K_Local2 = K_Local - K_Local1

        K_global1 = Rotation.T @ K_Local1 @ Rotation
        K_global2 = Rotation.T @ K_Local2 @ Rotation

        for h in range(2):
            if h%2==0:
                K_global = K_global1
                K_Global_GlobalCordi[(6*j):(6*j+6),(6*j):(6*j+6)] += K_global[0:6,0:6]
                K_Global_GlobalCordi[(6*j):(6*j+6),(6*k):(6*k+6)] += K_global[0:6,6:12]
                K_Global_GlobalCordi[(6*k):(6*k+6),(6*j):(6*j+6)] += K_global[6:12,0:6]
                K_Global_GlobalCordi[(6*k):(6*k+6),(6*k):(6*k+6)] += K_global[6:12,6:12]
            else:
                K_global = K_global2
                PartialRoatation[l:(l+3),l:(l+3)] = Rotation[3:6,3:6]*j_pinFlag + np.eye(3)*(1-j_pinFlag)
                PartialRoatation[m:(m+3),m:(m+3)] = Rotation[9:12,9:12]*k_pinFlag + np.eye(3)*(1-k_pinFlag)

                K_Global_GlobalCordi[(6*j):(6*j+3),(6*j):(6*j+3)] += K_global[0:3,0:3]#A
                K_Global_GlobalCordi[(6*j):(6*j+3),l    :(l+3)  ] += K_global[0:3,3:6]#B
                K_Global_GlobalCordi[(6*j):(6*j+3),(6*k):(6*k+3)] += K_global[0:3,6:9]#C
                K_Global_GlobalCordi[(6*j):(6*j+3),m    :(m+3)  ] += K_global[0:3,9:12]#D

                K_Global_GlobalCordi[l    :(l+3)  ,(6*j):(6*j+3)] += K_global[3:6,0:3]#E
                K_Global_GlobalCordi[l    :(l+3)  ,l    :(l+3)  ] += K_global[3:6,3:6]#F
                K_Global_GlobalCordi[l    :(l+3)  ,(6*k):(6*k+3)] += K_global[3:6,6:9]#G
                K_Global_GlobalCordi[l    :(l+3)  ,m    :(m+3)  ] += K_global[3:6,9:12]#H

                K_Global_GlobalCordi[(6*k):(6*k+3),(6*j):(6*j+3)] += K_global[6:9,0:3]#I
                K_Global_GlobalCordi[(6*k):(6*k+3),l    :(l+3)  ] += K_global[6:9,3:6]#J
                K_Global_GlobalCordi[(6*k):(6*k+3),(6*k):(6*k+3)] += K_global[6:9,6:9]#K
                K_Global_GlobalCordi[(6*k):(6*k+3),m    :(m+3)  ] += K_global[6:9,9:12]#L

                K_Global_GlobalCordi[m    :(m+3)  ,(6*j):(6*j+3)] += K_global[9:12,0:3]#M
                K_Global_GlobalCordi[m    :(m+3)  ,l    :(l+3)  ] += K_global[9:12,3:6]#N
                K_Global_GlobalCordi[m    :(m+3)  ,(6*k):(6*k+3)] += K_global[9:12,6:9]#O
                K_Global_GlobalCordi[m    :(m+3)  ,m    :(m+3)  ] += K_global[9:12,9:12]#P

            np.append(lmprint, [i, j, k, l, m]) # 확인용

    return PartialRoatation @ K_Global_GlobalCordi @ PartialRoatation.T

def pinIndexing(j, k, NodeNumber, pinNodeList, pinCount):
    j_pinFlag = (j in pinNodeList)*1
    k_pinFlag = (k in pinNodeList)*1

    l = int(6*j+3)*(1-j_pinFlag)+int(6*NodeNumber +3*pinCount)*j_pinFlag
    pinCount += j_pinFlag
    m = int(6*k+3)*(1-k_pinFlag)+int(6*NodeNumber +3*pinCount)*k_pinFlag
    pinCount += k_pinFlag

    return j_pinFlag, k_pinFlag, l, m, pinCount

def Local_K(j, k, Length_xyz, Elasticity, G_Shear, I_xyz, Area, pinNodeList):
    K_Local = np.zeros((12,12)) 

    K_Local[0,6] = -Elasticity[j]*Area[j]/Length_xyz
    K_Local[1,5] = 6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2
    K_Local[1,7] = -12*Elasticity[j]*I_xyz[j,2]/Length_xyz**3
    K_Local[1,11] = 6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2
    K_Local[2,4] = -6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2
    K_Local[2,8] = -12*Elasticity[j]*I_xyz[j,2]/Length_xyz**3
    K_Local[2,10] = -6*Elasticity[j]*I_xyz[j,2]/Length_xyz**2
    K_Local[3,9] = -G_Shear[j]*I_xyz[j,0]/Length_xyz
    K_Local[4,8] = 6*Elasticity[j]*I_xyz[j,2]/Length_xyz**2
    K_Local[4,10] = 2*Elasticity[j]*I_xyz[j,2]/Length_xyz
    K_Local[5,7] = -6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2
    K_Local[5,11] = 2*Elasticity[j]*I_xyz[j,1]/Length_xyz
    K_Local[7,11] = -6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2
    K_Local[8,10] = 6*Elasticity[j]*I_xyz[j,1]/Length_xyz**2

    K_Local += K_Local.T

    K_Local[0,0] = Elasticity[j]*Area[j]/Length_xyz
    K_Local[1,1] = 12*Elasticity[j]*I_xyz[j,1]/Length_xyz**3
    K_Local[2,2] = 12*Elasticity[j]*I_xyz[j,2]/Length_xyz**3
    K_Local[3,3] = G_Shear[j]*I_xyz[j,0]/Length_xyz
    K_Local[4,4] = 4*Elasticity[j]*I_xyz[j,2]/Length_xyz
    K_Local[5,5] = 4*Elasticity[j]*I_xyz[j,1]/Length_xyz
    K_Local[6,6] = Elasticity[j]*Area[j]/Length_xyz
    K_Local[7,7] = 12*Elasticity[j]*I_xyz[j,1]/Length_xyz**3
    K_Local[8,8] = 12*Elasticity[j]*I_xyz[j,2]/Length_xyz**3
    K_Local[9,9] = G_Shear[j]*I_xyz[j,0]/Length_xyz
    K_Local[10,10] = 4*Elasticity[j]*I_xyz[j,2]/Length_xyz
    K_Local[11,11] = 4*Elasticity[j]*I_xyz[j,1]/Length_xyz

    return K_Local

def RotationMat(j, k, Length, NodeGenCordi):
    cos1 = (NodeGenCordi[k,0] - NodeGenCordi[j,0])/Length
    cos2 = (NodeGenCordi[k,1] - NodeGenCordi[j,1])/Length
    cos3 = (NodeGenCordi[k,2] - NodeGenCordi[j,2])/Length

    D = (cos1**2+cos2**2)**0.5

    Rotation = np.zeros((12,12))

    for h in range(4):
        if D==0:
            if cos3>0:
                Rotation[3*h,3*h]=0
                Rotation[3*h,3*h+1]=0
                Rotation[3*h,3*h+2]=1
                Rotation[3*h+1,3*h]=0
                Rotation[3*h+1,3*h+1]=1
                Rotation[3*h+1,3*h+2]=0
                Rotation[3*h+2,3*h]=-1
                Rotation[3*h+2,3*h+1]=0
                Rotation[3*h+2,3*h+2]=0
            else:
                Rotation[3*h,3*h]=0
                Rotation[3*h,3*h+1]=0
                Rotation[3*h,3*h+2]=-1
                Rotation[3*h+1,3*h]=0
                Rotation[3*h+1,3*h+1]=1
                Rotation[3*h+1,3*h+2]=0
                Rotation[3*h+2,3*h]=1
                Rotation[3*h+2,3*h+1]=0
                Rotation[3*h+2,3*h+2]=0
        else:
            Rotation[3*h,3*h]=cos1
            Rotation[3*h,3*h+1]=cos2
            Rotation[3*h,3*h+2]=cos3
            Rotation[3*h+1,3*h]=-cos2/D
            Rotation[3*h+1,3*h+1]=cos1/D
            Rotation[3*h+1,3*h+2]=0
            Rotation[3*h+2,3*h]=-cos1*cos3/D
            Rotation[3*h+2,3*h+1]=-cos2*cos3/D
            Rotation[3*h+2,3*h+2]=D
    
    return Rotation




