# Guideline: https://builtin.com/software-engineering-perspectives/python-class-decorator

import pprint


# Decorate function with class method without params
class Power:
    def __init__(self, arg):
        self._arg = arg
        self._memory = []

    def __call__(self, a, b):
        result = self._arg(a, b) ** 2

        self._memory.append(result)
        return result

    def get_memory(self):
        return self._memory


@Power
def multiply(a, b):
    return a * b


print(
    multiply(2, 2),
    multiply(3, 3),
    multiply(4, 4)
)

print(multiply.get_memory())


# Decorate function with class method with params
class PowerParams:
    def __init__(self, arg):
        self._arg = arg

    def __call__(self, multiplier):

        def wrapper(a, b):
            result = multiplier(a, b)
            return result * self._arg
        return wrapper


@PowerParams(3)
def summ(a, b):
    return a + b


print(summ(1, 2))


class Logger:

    _journal = []

    def log(self, func):
        self._journal.append('Calling "{}". Result: {}'.format(func.__name__, func(self)))
        return func

    def get_journal(self):
        pprint.pp(self._journal)


logger = Logger()


class ClassToLog:

    @logger.log
    def walk(self):
        print('walking')

    @logger.log
    def ride(self):
        return 'riding result'


obj = ClassToLog()
obj.ride()

logger.get_journal()
