import numpy as np
import matplotlib.pyplot as plt
import random


curve_with_BN = np.loadtxt('curve_with_BN.txt')
curve_without_BN = np.loadtxt('curve_without_BN.txt')

a = 0.001
b = 0.739
print(len(curve_with_BN))

curve_with_BN_least = np.ones(len(curve_with_BN))
curve_without_BN_least = np.ones(len(curve_without_BN))
least = 1e10
for i in range(len(curve_with_BN)):
    if curve_with_BN[i] < least:
        least = curve_with_BN[i]
        curve_with_BN_least[i] = curve_with_BN[i]
    else:
        curve_with_BN_least[i] = least

least = 1e10
for i in range(len(curve_without_BN)):
    if curve_without_BN[i] < least:
        least = curve_without_BN[i]
        curve_without_BN_least[i] = curve_without_BN[i]
    else:
        curve_without_BN_least[i] = least

print(curve_without_BN_least[-1])



plt.figure(1)
plt.plot(range(len(curve_with_BN)),curve_with_BN,'steelblue',label='with Batch Normalization',linestyle='-')
plt.plot(range(len(curve_without_BN)),curve_without_BN,'orange',label='without Batch Normalization',linestyle='-')
plt.xscale('log')
plt.xlabel('Epochs')
plt.ylabel('Mean Square Error(validation set)')
plt.xlim([1000,1000000])
plt.ylim([0,4])

plt.figure(2)
plt.plot(range(len(curve_with_BN)),curve_with_BN_least,'orange',label='with Batch Normalization',linestyle='-',linewidth=3)
plt.plot(range(len(curve_without_BN)),curve_without_BN_least,'steelblue',label='without Batch Normalization',linestyle='-',linewidth=3)
plt.xscale('log')
plt.xlabel('Epochs')
plt.yscale('log')
plt.ylabel('Least Mean Square Error(validation set)')
plt.xlim([1000,1000000])
plt.ylim([10**-4,10**5])
plt.text(270000,0.0004,'(270545,0.001)',ha='center',)
plt.text(len(curve_without_BN),0.85,'(1000000,0.335)',ha='center',)
plt.legend()
plt.show()
