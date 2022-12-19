import numpy as np
print(np.__version__)

john = [10, 8, 6, 7]
x = np.array(john)
print(x)
print(type(x))

# Numpy vs. 일반 배열
john = [10, 8, 6, 7]
x = [c+1 for c in john]

john = np.array(john)
john + 1

# 다차원 배열
score = [[1,2,3,4],[5,6,7,8]]
X = np.array(score)
X.shape
X.dtype

# 1차원 배열
y = np.array([1.,2.,3.,4.])
y.shape
y.dtype

# list
list(range(1,10))
X = np.arange(0,16)

# Reshape
y = X.reshape(4,4)
y.shape
X = np.arange(0,16)
y = X.reshape(4,4)
y = X.reshape(2,4,2)
x = np.array([1,2,3]) # 1차원
y = np.array([[1],[2],[3]]) # 2차원

print(x.shape)
print(y.shape)

y.reshape(-1) #y.reshape(len(y))

X = np.arange(0,16)
X = X.reshape(-1,4,2)

# Slicing, Indexing
X = np.array([4,5,6,7,8,9,0])
X[:-1]
X[-1]

X = np.array([[4,5,6],[8,9,0]])
# fancy indexing
X[0,[1,2]]
X[0,1:]

X = np.array([1,2,3,4,5])
# boolean indexing
idx = [False, True, False, True,False]
X[idx]

X = np.array([1,5,3,7,6])
idx = X>6
X[idx] #X[X>6]

# Sum
X = np.array([[1,2,3],[3,4,5]])
np.sum(X, axis=0)
np.sum(X, axis=1)
np.sum(X)

# Mean
np.mean(X, axis=1)
np.mean(X, axis=0)
np.mean(X)

# Broadcasting
X = np.array([1,2,3])
X + 1

X = np.array([[1,2],[3,4]])
Y = np.array([[3,4],[5,6]]) 
np.add(X,Y)
np.subtract(X,Y)
np.divide(X,Y)

np.multiply(X,Y) # X@Y
np.dot(X,Y)
np.matmul(X,Y)

x = np.random.randn(5, 8)
np.argmax(x, axis=0)
np.argmax(x, axis=1)

# Random numbers (0~1, 1은 안나옴)
np.random.randn(2,4) # Return a sample (or samples) from the “standard normal” distribution
np.random.seed(0)
print(np.random.random(3))
np.random.seed(2)
print(np.random.random(3))
np.random.seed(0)
print(np.random.random(3))

# 저장 (savez())
x = np.random.randn(5, 4)
y= np.random.randn(3, 3)
print(x)
print(y)

np.savez("my_data.npz", xvar=x, yvar=y)

# 파일 로딩
my_data = np.load("my_data.npz")
x = my_data["xvar"]
print(x)