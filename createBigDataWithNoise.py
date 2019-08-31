import numpy as np
import matplotlib.pyplot as plt
import csv

def gaussian(amp, fwhm, mean):
    return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)
gausses = []
num = []
RMS = 0.005
NCHANNELS = 512
FILENAME = 'simple_gaussian.csv'
chan = np.arange(NCHANNELS)
numberOfOneGauss = 0
with open('gaussDataForNet.csv', mode='w') as csv_file:
    fieldnames = ['spec', 'numOfComp']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for z in range(0, 10000, 1):
        spectrum = np.random.randn(NCHANNELS) * RMS #np.zeros(NCHANNELS)
        numberOfComp = np.random.randint(low=0, high=3, size=1)
        for i in range(0, numberOfComp[0], 1):
            MEAN = np.random.randint(low=100, high=400, size=1)
            FWHM = np.random.randint(low=1, high=80, size=1)
            AMP = 1.0*np.random.randint(low=0, high=12, size=1)/12
            spectrum += gaussian(AMP, FWHM, MEAN)(chan)

        gausses.append(spectrum)
        num.append(numberOfComp[0])
        if (numberOfComp[0]==1):
            numberOfOneGauss+=1
        if z%1000 == 0:
            print z
        writer.writerow({'spec': ' '.join(map(str,spectrum)), 'numOfComp': numberOfComp[0]})
x_values = np.linspace(0, 512, 512)
print(numberOfOneGauss)
fig_gauss = plt.figure()
for i in range(1, 11, 1):
    ax_g = fig_gauss.add_subplot(2, 5, i)
    ax_g.set_title(num[i-1])

    ax_g.plot(x_values, gausses[i-1])
    plt.ylim(-0.1, 1)

plt.show()
