# Guideline: https://realpython.com/python-assert-statement/#understanding-pythons-assert-statements

# Asserting function's output
def f():
    return 3

assert f() == 10


# Asserting function's input
def f_2(input_var):
    assert input_var > 0 and isinstance(input_var, int), \
        f'Please use int larger than 10 here. Got {input_var} {type(input_var)}'
    print('Good!')

# f_2(17)


# Asserting variable's value
x = 10
assert x == 10

# Membership assertion
x_list = [1, 2, 3, 4]
assert 4 in x_list, '5 is not found'

# Identity assertion
x = 1
y = 2

assert x is not y
assert None is not x

assert all([True, True, True])
assert any([False, False, True])




