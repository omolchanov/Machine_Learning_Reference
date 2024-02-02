from sktime.datasets import load_arrow_head

arrow_X, arrow_y = load_arrow_head(return_type='numpy3d')

print(arrow_X)