# Guideline: https://www.programiz.com/python-programming/decorator
# Inner function inside the outer function
def outer(x):
    def inner(y):
        return x + y
    return inner


print(outer(5)(6))


# Passing the function as argument to another function
def add(x, y):
    return x + y


def calculate(func, x, y):
    return func(x, y)


print(calculate(add, 4, 6))


# Returns a function as a value
def greeting(name):
    def hello():
        return "Hello, " + name + "!"
    return hello


print(greeting('Vasya')())


# !! Basically, a decorator takes in a function, adds some functionality and returns it.
def make_pretty(func):
    def decorate():
        print("I got decorated")
        func()
    return decorate


@make_pretty
def ordinary():
    print("I am ordinary")


ordinary()


def decorate_divide(func):
    def decorate(a, b):
        assert b != 0, 'Zero division is not allowed'

        print('Dividing {} on {}'.format(a, b))
        return func(a, b)

    return decorate


def add_bias(func):
    def decorate(a, b):
        return func(a, b) + 3

    return decorate


@decorate_divide
@add_bias
def process(a, b):
    return a / b


print(process(2, 2))
