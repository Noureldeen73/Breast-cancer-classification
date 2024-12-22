import numpy as np
import pandas as pd

# arr1d = np.array([1, 2, 3])
# arr2d = np.array([[1, 2, 3], [4, 5, 6]])

# # print(arr1d[:, np.newaxis])  # arr[0][0] arr[1][0] arr[2][0]
# # print(arr1d[np.newaxis, :])  # arr[0][0] arr[0][1] arr[0][2]

# # arr[0][0] arr[1][0] arr[2][0] ----> arr[0] = [1, 2, 3] arr[1] = [4, 5, 6]
# print(arr2d[:, :, np.newaxis])  # arr[0][0][0] arr[0][1][0] arr[0][2][0]
# tryarray = arr2d[:, :, np.newaxis]
# print(tryarray.shape)
# print(tryarray[1][1][0])

df = pd.read_csv('./data/data.csv')

print(df.shape)

# i want to print the columns
print(df.columns)
