
# coding: utf-8

# # Convex optimization HW 3

# In[1]:

import numpy as np
import matplotlib.pyplot as plt


# ## Interior point method

# In[2]:

def phi(x, t, Q, p, A, b):
    n, r = Q.shape
    x = x.reshape(n,1)
    return t*((1/2)*np.dot(x.transpose(), np.dot(Q, x)) + np.dot(p.transpose(), x)) - np.sum(np.log(b - np.dot(A, x)))

def grad(x, t, Q, p, A, b):
    n, r = Q.shape
    u, v = A.shape
    x = x.reshape((n,1))
    
    denom = 1/(b - np.dot(A,x))
    grad = np.dot(A.transpose(), np.dot(np.diagflat(denom), np.ones((u,1))))
    
    return t*(np.dot(Q,x)+p) + grad

def hess(x, t, Q, p, A, b):
    n,r = Q.shape
    x = x.reshape(n,1)

    denom = 1/(b - np.dot(A,x))
    
    return t*Q + np.dot(A.transpose(),np.dot(np.diagflat(denom)**2, A))


# ## Newton method

# ### Question 1:

# In[3]:

def dampedNewtonStep(x, f, g, h):
    ag = g(x)
    invHess = np.linalg.inv(h(x))
    
    prodinvHess = np.dot(invHess, ag)
    newtonDecrement = np.dot(ag.transpose(), prodinvHess)
    
    return newtonDecrement/2, x - (1/(1+np.sqrt(newtonDecrement)))*prodinvHess


# ### Question 2:

# In[4]:

def dampedNewton(x0, f, g, h, tol):
    x = x0
    xhist = [x]
    gap, x = dampedNewtonStep(x, f, g, h)
    
    
    while gap > tol:
        gap, x = dampedNewtonStep(x, f, g, h)
        
        xhist.append(x)
        
    return x, xhist


# ### Question 3:

# In[5]:

def newtonLS(x0, f, g, h, tol):
    x = x0
    xhist = [x]
    invh = np.linalg.inv(h(x))
    Grad = g(x)
    gap = np.dot(Grad.transpose(), np.dot(invh, Grad))/2
    s = 1/4
    beta = 1/10
    
    while gap > tol:
        invh = np.linalg.inv(h(x))
        Grad = g(x)
        xnt = -np.dot(invh, g(x))
        t = 1/50 #We choose here a small enough t to stay in the feasible set, otherwise we have to add a parameter
        
        while f(x+ t*xnt) > f(x) + t*s*np.dot(Grad.transpose(), xnt):
            t *= beta
            
        x = x + t*xnt
        
        gap = np.dot(Grad.transpose(), -xnt)/2
        
        xhist.append(x)
        
    return x, xhist


# ## Support Vector Machine Problem

# ### Question 1:

# In[6]:

def transform_svm_primal(tau, X, y):
    n,r = X.shape
    
    p = (1/(n*tau))*np.concatenate((np.zeros((r,1)), np.ones((n,1))))
    b = np.concatenate((-np.ones((n, 1)), np.zeros((n, 1))))
    
    A1 = np.concatenate((X*y.reshape(n,1), np.eye(n)), axis=1)
    A2 = np.concatenate((np.zeros((n,r)), np.eye(n)), axis=1)
    A = -np.concatenate((A1, A2), axis=0)
    
    Q1 = np.concatenate((np.eye(r), np.zeros((r,n))), axis=1)
    Q2 = np.concatenate((np.zeros((n,r)), np.zeros((n,n))), axis=1)
    Q = np.concatenate((Q1, Q2), axis=0)
    
    
    return Q, p, A, b

def transform_svm_dual(tau, x, y):
    
    n,p = x.shape
    
    b = np.concatenate((1/(n*tau)*np.ones((n,1)), np.zeros((n,1))), axis=0)
    
    p = -np.ones((n,1))
    
    A = np.concatenate((np.eye(n), -np.eye(n)), axis=0)
    
    Q = np.array([[y[i]*y[j]*np.dot(x[i],x[j]) for i in range(n)] for j in range(n)])
    
    return Q, p, A, b


# ### Question 2

# In[7]:

def barr_method(Q, p, A, b, x0, mu, tol):
    t = 1
    x = x0

    m, r = A.shape
    xhist = [x]
    
    while m/t > tol:

        
        f = lambda x : phi (x, t, Q, p, A, b ) ;
        g = lambda x : grad (x, t, Q, p, A, b ) ;
        h = lambda x : hess (x, t, Q, p ,A, b ) ;
        
        x, a = dampedNewton(x, f, g, h, 1e-2)
    
        xhist.append(x)
        
        t = mu*t
        
    return xhist, x


# ### Question 3

# In[8]:

train = np.genfromtxt("iris.data", delimiter=',', names=("sepal1", "sepal2", "petal1", "petal2", "type"), dtype=None)
train = train[(train["type"] == b"Iris-versicolor") + (train["type"] == b"Iris-virginica")]
n = len(train)
y = (train["type"] == b"Iris-versicolor")*1
y[y==0] = -1
train = train[np.array(["petal1", "petal2", "sepal1", "sepal2"])].reshape(n,1)
train.dtype='float64'
train = train - np.repeat(np.mean(train, axis=0).reshape(1,4), n, axis=0) #We center the data


# In[9]:

tau = 1/2
x = train
n, r = x.shape
Q,p,A,b = transform_svm_primal(tau, x, y)
x0=np.concatenate((np.zeros((r,1)),np.ones((n,1))*2))
a = barr_method(Q, p, A, b, x0, 10, 1e-1)[1]

w = a[0:4]

print("Le vecteur w optimal est alors: \n", w)


# ### Question 4

# In[10]:

tau = 1/2
x = train
n, r = x.shape
x0 = (1/(2*n*tau))*np.ones((n,1))

indices = np.random.choice(range(100), 80, replace=False)


# In[11]:

#On essai d'évaluer la performance de l'algorithme en fonction de la valeur de tau

listtau = list(np.linspace(1e-2,3, 30))
    
    
x = train[indices]
n,r = x.shape
labels = y[indices]

reussite = []

for tau in listtau:
    resultatsOntau = []
    
    for i in range(5): #On effectue une moyenne de performance pour chaque tau
        indices = np.random.choice(np.arange(100), 80, replace=False)
        Q,p,A,b = transform_svm_primal(tau, x, labels)
        x0=np.concatenate((np.zeros((r,1)),np.ones((n,1))*2))
        a = barr_method(Q, p, A, b, x0, 10, 1e-1)[1]
        w = a[0:4]

        indicesTest = list(range(100))

        for elm in indices:
            indicesTest.remove(elm)


        indicesTest = np.array(indicesTest)
        xtest = train[indicesTest]
        ytest = y[indicesTest]
        res = (np.dot(xtest, w)>0)*1
        res[res==0] = -1
        
        resultatsOntau.append(sum(res==ytest.reshape((20,1)))[0]/20)
        
    reussite.append(sum(resultatsOntau)/5)

plt.plot(listtau, reussite)
plt.title("Performance moyenne de l'algorithme en fonction de la valeur de tau.")
plt.show()

print("Le classifieur semble meilleur pour tau très petit.")


# In[12]:

#Duality gaps with damped newton method:


#On the primal:
def dampedNewton2(x0, f, g, h, tol):
    x = x0
    gap, x = dampedNewtonStep(x, f, g, h)
    i = 0
    
    while gap > tol:
        gap, x = dampedNewtonStep(x, f, g, h)
        
        i += 1
        
    return x, i

def barr_method2(Q, p, A, b, x0, mu, tol):
    t = 1
    x = x0

    m, r = A.shape
    thist = [m]
   
    while m/t > tol:

        
        f = lambda x : phi (x, t, Q, p, A, b ) ;
        g = lambda x : grad (x, t, Q, p, A, b ) ;
        h = lambda x : hess (x, t, Q, p ,A, b ) ;
        
        x, i = dampedNewton2(x, f, g, h, 1e-2)
    
        
        t = mu*t
        
        thist += [m/t]*i
        
    return thist, x

tau = 1/2
x = train
n, r = x.shape
Q,p,A,b = transform_svm_primal(tau, x, y)

plts = []
legends = []
for mu in [2, 15, 50, 100]:
    
    x0=np.concatenate((np.zeros((r,1)),np.ones((n,1))*2))
    results = barr_method2(Q, p, A, b, x0, mu, 1e-3)

    plt.semilogy(results[0])
    legends.append("mu = {}".format(mu))


plt.legend(legends)
plt.title('Duality gaps vs Iterations for the damped newton, on the primal')
plt.show()
plt.close()



# In[13]:

#On the dual:

tau = 1/2
x = train
n, r = x.shape
Q,p,A,b = transform_svm_dual(tau, x, y)

plts = []
legends = []

for mu in [2, 15, 50, 100]:
    
    x0=(1/(2*n*tau))*np.ones((n,1))
    results = barr_method2(Q, p, A, b, x0, mu, 1e-3)

    plt.semilogy(results[0])
    legends.append("mu = {}".format(mu))


plt.legend(legends)
plt.title('Duality gaps vs Iterations for the damped newton, on the dual')
plt.show()
plt.close()


# In[14]:

#Duality gaps with linesearch newton method:

def newtonLS2(x0, f, g, h, tol, A, b): #We add 2 arguments to check if the points we get are feasible
    x = x0
   
    invh = np.linalg.inv(h(x))
    Grad = g(x)
    gap = np.dot(Grad.transpose(), np.dot(invh, Grad))/2
    s = 1/4
    beta = 1/10
    i = 0
    
    while gap > tol:
        invh = np.linalg.inv(h(x))
        Grad = g(x)
        xnt = -np.dot(invh, g(x))
        t = 1
        
        while np.sum((b - np.dot(A,x + t*xnt)) <= 0) > 0:
            t *= beta
        
        while f(x+ t*xnt) > f(x) + t*s*np.dot(Grad.transpose(), xnt):
            t *= beta
            
        x = x + t*xnt
        
        gap = np.dot(Grad.transpose(), -xnt)/2 
        
        i += 1
        
    return x, i


def barr_method3(Q, p, A, b, x0, mu, tol):
    t = 1
    x = x0

    m, r = A.shape
    thist = [m]
   
    while m/t > tol:

        
        f = lambda x : phi (x, t, Q, p, A, b ) ;
        g = lambda x : grad (x, t, Q, p, A, b ) ;
        h = lambda x : hess (x, t, Q, p ,A, b ) ;
        
        x, i = newtonLS2(x, f, g, h, 1e-2, A, b)
    
        
        t = mu*t
        
        thist += [m/t]*i
        
    return thist, x

tau = 1/2
x = train
n, r = x.shape
Q,p,A,b = transform_svm_primal(tau, x, y)

plts = []
legends = []
for mu in [2, 15, 50, 100]:
    
    x0=np.concatenate((np.zeros((r,1)),np.ones((n,1))*2))
    results = barr_method3(Q, p, A, b, x0, mu, 1e-3)

    plt.semilogy(results[0])
    legends.append("mu = {}".format(mu))


plt.legend(legends)
plt.title('Duality gaps vs Iterations for the newton linesearch, on the primal')
plt.show()
plt.close()


# In[15]:

#On the dual:

tau = 1/2
x = train
n, r = x.shape
Q,p,A,b = transform_svm_dual(tau, x, y)

plts = []
legends = []

for mu in [2, 15, 50, 100]:
    
    x0=(1/(2*n*tau))*np.ones((n,1))
    results = barr_method3(Q, p, A, b, x0, mu, 1e-3)

    plt.semilogy(results[0])
    legends.append("mu = {}".format(mu))


plt.legend(legends)
plt.title('Duality gaps vs Iterations for the newton linesearch, on the dual')
plt.show()
plt.close()

