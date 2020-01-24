

import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors

# %%

f = lambda x: -np.log2(x)
fp = lambda x: -1/x/np.log(2)

x = np.hstack([np.linspace(0.0001, .001, 10), np.linspace(0.001, .01, 10), np.linspace(0.01, .1, 20), np.linspace(.1,1,20)])
plt.figure(figsize=(12, 4))
plt.plot(x, f(x), label="- log($p_{y_i}$)")
plt.plot(x, fp(x), label=" $\\nabla$ -log($p_{y_i}$)")
x = np.array([.8, .2, .5])
plt.scatter(x, f(x))
plt.scatter(x, fp(x))
plt.grid()
plt.tight_layout()
plt.legend()
plt.ylim(-8, 10)
plt.savefig('mle-grad.svg')
plt.show()

# %%
print(fp(np.array([0.2, 0.5, 0.8])))
print(fp(1))

# %%

def bayes_risk(k=1, n_p=10, ratio=9, n_n=None, plot=True, show=True):
    if n_n is None: n_n = n_p * ratio
    Xp = np.random.uniform(0, 1, (n_p, 2))
    Xm = np.random.uniform(0, 1, (n_n, 2))
    X = np.vstack([Xp, Xm])
    Y = np.array([1]*Xp.shape[0] + [0]*Xm.shape[0])
    if plot:
        plt.scatter(Xm[:,0], Xm[:,1], color='blue', alpha=0.4, marker='_')
        plt.scatter(Xp[:,0], Xp[:,1], color='green', marker='+')

    nn = sklearn.neighbors.KNeighborsClassifier(k)
    nn = nn.fit(X, Y)

    l = lambda n: np.linspace(0, 1, n)
    xx, yy = np.meshgrid(l(100), l(100))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if plot:
        plt.contour(xx, yy, Z, levels=[0.5])
        if show:
            plt.show()

    return Z.mean()

    #test = np.random.uniform(0, 1, (100000, 2))
    #dists, inds = nn.kneighbors(test)
    #print('Expected:', Y.mean(), 'empirical value:', Y[inds[:,0]].mean())

# %% LONG LONG

ks = [1,3,5]
n_ps = [10, 50, 100, 200, 1000]
runs = list(range(10))

res = np.zeros((len(ks), len(n_ps), len(runs)))
for ik,k in enumerate(ks):
    for in_p,n_p in enumerate(n_ps):
        for ir,r in enumerate(runs):
            res[ik,in_p,ir] = bayes_risk(k, n_p, plot=False)


# %%
#plt.imshow(res.mean(axis=2))
#plt.colorbar()
plt.plot(res[0,:,:].mean(axis=1))
plt.plot(res[1,:,:].mean(axis=1))
plt.plot(res[2,:,:].mean(axis=1))

# %%
XXp = np.random.uniform(0, 0.5, (10000, 2)) * np.array([[1, 2]])
XXm = np.random.uniform(0, 0.5, (10000, 2)) * np.array([[1, 2]]) + np.array([[0.5, 0]])
def separated(k=1, n_p=10, ratio=9, n_n=None, plot=True, show=True, save=False, name=None):
    if n_n is None: n_n = n_p * ratio
    Xp = XXp[:n_p,:]
    Xm = XXm[:n_n,:]
    X = np.vstack([Xp, Xm])
    Y = np.array([1]*Xp.shape[0] + [0]*Xm.shape[0])
    if plot:
        plt.scatter(Xm[:,0], Xm[:,1], color='blue', alpha=0.4, marker='_')
        plt.scatter(Xp[:,0], Xp[:,1], color='green', marker='+')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    nn = sklearn.neighbors.KNeighborsClassifier(k)
    nn = nn.fit(X, Y)

    #test = np.random.uniform(0, 1, (100000, 2))
    #dists, inds = nn.kneighbors(test)
    #print('Expected:', 0.5, 'empirical value:', Y[inds[:,0]].mean())

    l = lambda n: np.linspace(0, 1, n)
    xx, yy = np.meshgrid(l(100), l(100))
    Z = nn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if plot:
        plt.tight_layout()
        plt.contour(xx, yy, Z, levels=[0.5])
        if save:
            if name is None:
                name = 'knn-boundary-%d-%d-%d'%(k, n_p, ratio)
            plt.savefig(name+'.svg')
            plt.savefig(name+'.png')
        if show:
            plt.show()

for k in [1,5]:
    for n_p in [10, 100, 1000]:
        for ratio in [10, 100]:
            separated(k, n_p, ratio, save=True)

separated(11, 100, 10, save=True)
separated(11, 20, 10, save=True)
separated(11, 10, 10, save=True)




























#
