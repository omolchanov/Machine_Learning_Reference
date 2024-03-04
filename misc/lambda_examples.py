# Guideline: https://habr.com/ru/companies/piter/articles/674234/

from functools import reduce

# Simple lambda function
print((lambda x: x * 2)(5))
print((lambda x, y: x ** y)(2, 3))

# Lamda as an argument in another function
my_list = [1, 3, 4, 6, 10, 11, 15, 12, 14]

new_list = list(filter(lambda x: (x % 2 == 0), my_list))
print(new_list)

new_list = list(map(lambda x: x * 2, my_list))
print(new_list)

sum_list = reduce((lambda x, y: x + y), new_list)
print(sum_list)

# Lists with lambda function
tables = [lambda x=x: x for x in range(10)]
for i in tables:
    print(i())

# Short If and lambda functions
print((lambda a, b: a if a > b else b)(10, 5))
