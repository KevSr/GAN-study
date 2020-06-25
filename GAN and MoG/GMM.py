import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.stats as stats
from PIL import Image
import gzip
import tensorflow as tf

lower, upper = 0, 255
r=0

def MMISEL_R():
    m1, s1 = 50, 20
    m2, s2 = 200, 60
    low1, up1 = (lower - m1)/ s1 , (upper - m1)/ s1
    low2, up2 = (lower - m2)/ s2 , (upper - m2)/ s2
    
    s1 = stats.truncnorm(low1, up1, m1, s1).rvs(1000)
    s2 = stats.truncnorm(low2, up2, m2, s2).rvs(1000)

    s = np.append(s1, s2)

    return s

def MMISEL_GB():
    m1, s1 = 50, 20
    m2, s2 = 150, 60
    low1, up1 = (lower - m1)/ s1 , (upper - m1)/ s1
    low2, up2 = (lower - m2)/ s2 , (upper - m2)/ s2

    s1 = stats.truncnorm(low1, up1, m1, s1).rvs(1000)
    s2 = stats.truncnorm(low2, up2, m2, s2).rvs(1000)

    s = np.append(s1, s2)

    return s

def KDEF_R():
    m1, s1 = 10, 5
    m2, s2 = 90, 90
    m3, s3 = 150, 10
    low1, up1 = (lower - m1)/ s1 , (upper - m1)/ s1
    low2, up2 = (lower - m2)/ s2 , (upper - m2)/ s2
    low3, up3 = (lower - m3)/ s3 , (upper - m3)/ s3
    

    s1 = stats.truncnorm(low1, up1, m1, s1).rvs(200)
    s2 = stats.truncnorm(low2, up2, m2, s2).rvs(1000)
    s3 = stats.truncnorm(low3, up3, m3, s3).rvs(600)

    s = np.append(s1, s2)
    s = np.append(s, s3)

    return s

def KDEF_G():
    m1, s1 = 10, 5
    m2, s2 = 70, 30
    m3, s3 = 140, 10
    low1, up1 = (lower - m1)/ s1 , (upper - m1)/ s1
    low2, up2 = (lower - m2)/ s2 , (upper - m2)/ s2
    low3, up3 = (lower - m3)/ s3 , (upper - m3)/ s3
    

    s1 = stats.truncnorm(low1, up1, m1, s1).rvs(400)
    s2 = stats.truncnorm(low2, up2, m2, s2).rvs(1000)
    s3 = stats.truncnorm(low3, up3, m3, s3).rvs(800)

    s = np.append(s1, s2)
    s = np.append(s, s3)

    return s   

def KDEF_B():
    m1, s1 = 5, 5
    m2, s2 = 50, 15
    m3, s3 = 100, 10
    low1, up1 = (lower - m1)/ s1 , (upper - m1)/ s1
    low2, up2 = (lower - m2)/ s2 , (upper - m2)/ s2
    low3, up3 = (lower - m3)/ s3 , (upper - m3)/ s3
    

    s1 = stats.truncnorm(low1, up1, m1, s1).rvs(500)
    s2 = stats.truncnorm(low2, up2, m2, s2).rvs(800)
    s3 = stats.truncnorm(low3, up3, m3, s3).rvs(1000)

    s = np.append(s1, s2)
    s = np.append(s, s3)

    return s

def inverse_transform_sampling(data, n_bins=40, n_samples=1000, seed = 1):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    cum_values = np.zeros(bin_edges.shape)
    cum_values[1:] = np.cumsum(hist*np.diff(bin_edges))
    inv_cdf = interpolate.interp1d(cum_values, bin_edges)
    r = tf.random.uniform(shape = [n_samples], seed = seed)
    return inv_cdf(r), cum_values

def produce_train_set(model):
    x, cum = inverse_transform_sampling(model, n_samples=3920000)
    train = np.array(loop(x))
    print(train.shape)
    return train

def produce_valid_set(model):
    x, cum = inverse_transform_sampling(model, n_samples=3920000, seed = 2)
    train = np.array(loop(x))
    print(train.shape)
    return train

def produce_test_set(num, model):
    test = 0
    for i in range(num):
        x, cum = inverse_transform_sampling(model, n_samples=7840, seed = 2+i)
        train = np.array(loop(x, maxi=7840))
        if i == 0:
            test = train
        else:
            test = np.append(test, train, axis= 0)
        print(test.shape)
    return test

def loop(x, maxi =3920000):
    train = []
    n=0
    for k in range(len(x)):
        z = np.ndarray(shape=(28,28))
        for i in range(28):
            for j in range(28):
                z[i, j] = x[n]
                n += 1
        train.append(z)
        if n == maxi:
            print('done')
            break
    return train

def all_train_set():
    np.save('MMISEL_R_train.npy', produce_train_set(MMISEL_R()))
    np.save('MMISEL_GB_train.npy', produce_train_set(MMISEL_GB()))
    np.save('KDEF_R_train.npy', produce_train_set(KDEF_R()))
    np.save('KDEF_G_train.npy', produce_train_set(KDEF_G()))
    np.save('KDEF_B_train.npy', produce_train_set(KDEF_B()))

def all_valid_set():
    np.save('MMISEL_R_valid.npy', produce_valid_set(MMISEL_R()))
    np.save('MMISEL_GB_valid.npy', produce_valid_set(MMISEL_GB()))
    np.save('KDEF_R_valid.npy', produce_valid_set(KDEF_R()))
    np.save('KDEF_G_valid.npy', produce_valid_set(KDEF_G()))
    np.save('KDEF_B_valid.npy', produce_valid_set(KDEF_B()))

def all_test_set():
    np.save('MMISEL_R_test.npy', produce_test_set(50, MMISEL_R()))
    np.save('MMISEL_GB_test.npy', produce_test_set(50, MMISEL_GB()))
    np.save('KDEF_R_test.npy', produce_test_set(50, KDEF_R()))
    np.save('KDEF_G_test.npy', produce_test_set(50, KDEF_G()))
    np.save('KDEF_B_test.npy', produce_test_set(50, KDEF_B()))

all_test_set()
# plt.show()
# plt.hist(x, bins = 128)
# plt.title("Sample histogram for KDEF_B")
# plt.savefig('KDEF_B_inv.png')
# plt.show()
