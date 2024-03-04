# Iterator - saves values in memory
iterator = [x for x in range(10)]
print(iterator)

# Generator - does not save objects in memeory, they can be read once
generator = (x for x in range(10))
for i in generator:
    print(i)


# The yield example
def create_generator(my_list):
    for i in my_list:
        yield i ** 3


gen = create_generator((x for x in range(15)))
for i in gen:
    print(i)


# Yield in function
# Return sends a specified value back to its caller whereas Yield can produce a sequence of values.
def yield_func():
    yield 1
    yield 2
    yield 3


for i in yield_func():
    print(i)
