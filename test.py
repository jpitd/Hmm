import numpy as np



if __name__=="__main__":
    a=np.arange(4).reshape(4,1)
    b=np.array([[1,2,8,4],[5,6,7,8]])
    c=np.arange(4)
    print(np.where(b[:,3:4]==8)[0])
