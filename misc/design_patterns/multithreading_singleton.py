# Guidelines
# https://refactoring.guru/ru/design-patterns/singleton
# https://refactoring.guru/ru/design-patterns/singleton/python/example

from threading import Lock, Thread


class SingletonMeta(type):
    """
    A thread-safe implementation of the Singleton class.
    """

    _instances = {}

    # a Lock object to synchronize threads during first access to Singleton.
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            """
            The first thread reaches this condition and goes inside, creating a Singleton object. 
            As soon as this thread leaves the section and leaves the lock, the next thread can 
            acquire the lock again and go inside.
            However, now the Singleton instance will already be created and the thread will not be 
            able to pass through this condition, which means the new object will not will be created.
            """

            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance

            return cls._instances[cls]


class Singleton:

    value: str = None

    def __init__(self, value: str) -> None:
        self.value = value

    def get_id(self):
        return id(self)


def test_singleton(value: str) -> None:
    """
    A client code to test the Singleton logic
    :param value: string
    """

    st = Singleton(value)
    print(st.value)
    print(st.get_id())


if __name__ == '__main__':
    thr_1 = Thread(target=test_singleton, args=('Obj 1', ))
    thr_2 = Thread(target=test_singleton, args=('Obj 2',))

    thr_1.start()
    thr_2.start()
