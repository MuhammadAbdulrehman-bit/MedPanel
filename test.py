# # import numpy as np

# my_array = np.array(
#     [1,2,3,4,5,6,7] 
# )

# my_array_2 = np.array(
#    [         #column 
#        [1,2,3,4], # 1,4,5 = 1x4x5 = 20 = 20
#     [5,6,7,8]
#    ]# (1,3,5,2): Outer array encloses 1 array. 
# )
# print(my_array_2.shape)

# print(my_array_2[0:3 , 1:3])

import pandas as pd

dictionary1 = {
    "Name" : ["Alice", "Bob", "Charlie"],
    "Age" : [25, 30, 35],
    "City" : ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(dictionary1)
print(df["Name"])
