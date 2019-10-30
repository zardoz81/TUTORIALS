#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:58:02 2019

@author: romankoshkin
"""

H = [0,7,8,9]
U1 =  [3,4,5]
X =  [6,3,2]
x = 4


#H = [0]
#U1 =  []

dLdu1 = 1.5
Wr = 1

t_dldWr = 0
t_dldW1 = 0



dLdWr = dLdu1 * H[-2]
for j in reversed(range(len(U1)-1)):
    tmp1 = dLdu1
    for u1 in U1[j:-1]:
        tmp1 *= Wr * u1
    tmp1 *= H[j]
    dLdWr += tmp1
t_dldWr += dLdWr


dLdW1 = dLdu1 * X[-1]
for j in reversed(range(len(U1)-1)):
    tmp1 = dLdu1
    for u1 in U1[j:-1]:
        tmp1 *= Wr * u1
    tmp1 *= X[j]
    dLdW1 += tmp1
t_dldW1 += dLdW1

    

#        print('U1: ', U1[t],' H: ', H[c], ' t: ', t, ' c: ', c)
#        tmp1 *= Wr * U1[t]
#    dLdWr += tmp1 * H[c]