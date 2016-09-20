# encoding:utf-8

import  numpy as np


num=np.zeros(10)
# num[1]=10
print(num)
print(num==num.max())
ans=np.where(num==num.max())
print(len(ans[0]))

print(5%3)
