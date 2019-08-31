from matplotlib import pyplot as mp
import numpy as np
import csv

def gaussian(x, amp, fwhm, mean):
    return amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)

x_values = np.linspace(0, 164, 164)
with open('gaussDataForNet.csv', mode='w') as csv_file:
    fieldnames = ['spec', 'numOfComp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for z in range(0, 10, 1):
        gauss = np.zeros(164)
        numberOfComp = np.random.randint(low=1, high=10, size=1)
        for i in range(0, numberOfComp[0], 1):
            mean = np.random.randint(low=1, high=164, size=1)
            fwhm = np.random.randint(low=1, high=80, size=1)
            amp = np.random.randint(low=1, high=12, size=1)
            gauss += gaussian(x_values, amp[0], fwhm[0], mean[0])
        writer.writerow({'spec': gauss, 'numOfComp': numberOfComp[0]})
        #if z/1000 == 0:
        print (gauss)
'''print(numberOfComp[0])
mp.plot(x_values, gauss)
mp.show()'''