import numpy as np

def write_to_csv(*args, filename, header=None):
    # create a list that will contain the data
    data = []
    # loop over the arrays
    for arg in args:
        # append the array to the list
        data.append(arg)
    # transpose the list
    data = np.array(data).T
    # write the data to a csv file
    if header is not None:
        np.savetxt(f'{filename}.csv', data, delimiter=',', header=header)
    else:
        np.savetxt(f'{filename}.csv', data, delimiter=',')
    return None

