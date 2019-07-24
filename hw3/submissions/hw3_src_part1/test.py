import matlab.engine
from scipy.io import loadmat
import pandas as pd


# eng = matlab.engine.start_matlab()
# eng.p1_multi_layers(nargout=0)

apple1 = loadmat('./apples.mat')
apple1 = pd.DataFrame(apple1["apples"])
apple2 = loadmat('./apples2.mat')
apple2 = pd.DataFrame(apple2["apples2"])

print(apple1)
