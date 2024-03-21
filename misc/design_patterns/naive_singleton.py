# Guidelines
# https://refactoring.guru/ru/design-patterns/singleton
# https://refactoring.guru/ru/design-patterns/singleton/python/example


class SingletonMeta(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):

    def get_id(self):
        return id(self)


obj1 = Singleton()
obj2 = Singleton()

print(obj1 == obj2)

print(obj1.get_id())
print(obj2.get_id())
