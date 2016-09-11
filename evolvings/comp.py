#!/usr/bin/env python
"""
Compute the coherence of two signals
"""
import numpy as np
import matplotlib.pyplot as plt

# make a little extra space between the subplots
plt.subplots_adjust(wspace=0.5)

dt = 0.01
t = np.arange(0, 30, dt)
nse1 = np.random.randn(len(t))                 # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2
r = np.exp(-t/0.05)

# Hourly results
real = np.loadtxt("data/daily")
pers = np.loadtxt("data/bench_daily")
krls = np.loadtxt("2.txt")

plt.plot(real)
plt.plot(krls)
plt.show()
print len(real), len(pers), len(krls)

plt.subplot(211)
plt.plot(real[4:] - krls, 'g-')
#plt.plot(s2, 'g-')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)

plt.subplot(212)
plt.plot(real[1:] - pers, 'b-')
#plt.plot(krls - pers[2:], 'b-')
#cxy, f = plt.cohere(s1, s2, 256, 1./dt)
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
plt.xlabel('Time')
plt.show()