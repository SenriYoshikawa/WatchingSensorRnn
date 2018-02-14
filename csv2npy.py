import sys
import numpy as np
import pickle
import pandas as pd

for i in range(1, len(sys.argv)) :

	df = pd.read_csv(sys.argv[i])
	np_df = np.array(df)

	data_np = np_df[:, 2:4]
	data_np[data_np == '1'] = 1
	data_np[data_np == '2'] = 2
	data_np[data_np == '3'] = 3
	data_np[data_np == '4'] = 4
	data_np[data_np == '5'] = 5
	data_np[data_np == '6'] = 6
	data_np[data_np == '7'] = 7
	data_np[data_np == '8'] = 8
	data_np[data_np == '9'] = 9
	data_np[data_np == 'A'] = 10
	data_np[data_np == 'B'] = 11
	data_np[data_np == 'C'] = 12
	data_np[data_np == 'D'] = 13
	data_np[data_np == 'E'] = 14
	data_np[data_np == 'F'] = 15
	data_np[data_np == 'x'] = -1
	data_np[data_np == 'X'] = -1
	data_np[data_np == ''] = -1

	print(data_np)
	np.save(sys.argv[i][:-3] + 'npy', data_np)

