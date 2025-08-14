import numpy as np 

def get_grid_layout(n_subplots): 
    assert isinstance(n_subplots, int), 'Number of subplots must be an integer'
    assert n_subplots > 0, 'Number of subplots cannot be less than 1'

    is_square = lambda x: np.pow(int(np.sqrt(x)), 2) == x

    if is_square(n_subplots): 
        n_rows = n_cols = int(np.sqrt(n_subplots))
    elif n_subplots % 2 == 0: 
        n_cols = 2
        n_rows = int(n_subplots // n_cols)
    elif n_subplots % 3 == 0 :
        n_cols = 3
        n_rows = int(n_subplots // n_cols)
    else: 
        n_rows = int(np.ceil(np.sqrt(n_subplots)))
        n_cols = int(np.ceil(n_subplots / n_rows))
    
    return n_rows, n_cols