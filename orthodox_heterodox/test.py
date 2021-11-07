import numpy as np

random_prob = np.random.rand(10000)
random_labels = np.random.rand(10000)>0.9
random_mse_pos = np.mean((random_labels[random_labels==1] - random_prob[random_labels==1])**2)
random_mse_neg = np.mean((random_labels[random_labels!=1] - random_prob[random_labels!=1])**2)
print(random_mse_pos)
print(random_mse_neg)



