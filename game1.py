import numpy as np
random_arr=np.random.random((1,3))
print(random_arr)
if (random_arr[0,0] == random_arr[0,1] == random_arr[0,2]):
    print("LUCKY")
else:
    print("try again")