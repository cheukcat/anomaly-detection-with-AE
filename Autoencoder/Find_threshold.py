# Is NN a better way than setting a threshold?
import os
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------#
# Loading data
#---------------------------------#
tempStore = os.path.join('../', 'tempData')

def L2_loss(Truth, Prediction):
    return np.sum(np.square(Truth - Prediction), axis=1)

normal_origin = np.load(os.path.join(tempStore,'normal_origin.npy'))
attack_origin = np.load(os.path.join(tempStore,'attack_origin.npy'))
normal_recon = np.load(os.path.join(tempStore,'normal_recon.npy'))
attack_recon = np.load(os.path.join(tempStore,'attack_recon.npy'))

L2_normal = L2_loss(normal_origin, normal_recon)
L2_attack = L2_loss(attack_origin, attack_recon)

# Take full batch while using threshold


thresholds = np.linspace(0, 1.1 , 550)
acc_normal = []
acc_attack = []
for threshold in thresholds:
    acc_normal.append(np.sum((L2_normal < threshold).astype(float)) / len(L2_normal))
    acc_attack.append(np.sum((L2_attack > threshold).astype(float)) / len(L2_attack))
acc = (np.array(acc_attack) + np.array(acc_normal)) / 2

plt.figure()
plt.plot(thresholds, acc, 'r', thresholds, acc_normal, 'g', thresholds, acc_attack, 'b')
plt.ylabel('Thresholds')
plt.ylabel('Acc')
plt.show()

plt.figure()
plt.plot(thresholds[:30], acc[:30], 'r', thresholds[:30], acc_normal[:30], 'g', thresholds[:30], acc_attack[:30], 'b')
plt.ylabel('Thresholds')
plt.ylabel('Acc')
plt.show()