# Guidelines:
# https://refactoring.guru/ru/design-patterns/proxy
# https://refactoring.guru/ru/design-patterns/proxy/python/example

import datetime
from abc import ABC, abstractmethod


class Subject(ABC):
    """
    The Subject Interface declares common operations for both the Real Subject  and the Proxy.
    While the client is working with the Real Subject using this interface, you can pass it
    a Proxy instead of a real subject.
    """

    @abstractmethod
    def request(self) -> str:
        pass


class RealSubject(Subject):
    """
    The Real Subject class contains some basic business logic.
    """

    def request(self) -> str:
        print('Real Subject: Handling request')


class Proxy(Subject):
    """
    The Proxy's interface is identical to that of the Real Subject.
    """

    def __init__(self, real_subject: RealSubject) -> None:
        self._real_subject = real_subject

    def request(self) -> str:
        """
        The most common areas of application of the Proxy pattern are lazy loading, caching,
        access control, logging, etc.
        The Proxy can perform one of these tasks and then, depending on the result, transfer
        execution to the method of the same name in associated object of the Real Subject class.
        """

        if self.check_access() is True:
            self._real_subject.request()
            self.logging()

    def check_access(self) -> bool:
        print('Checking access before doing a real request...')
        return True

    def logging(self) -> None:
        print('Logging the time of request: ', datetime.datetime.now())


def client_code(subject: Subject):
    """
    The Client code must work with all objects (both real and proxies) through the Subject interface
    to maintain real subjects and proxies.
    """

    return subject.request()


if __name__ == '__main__':
    print('Client: Executing the client code with a real subject:')
    real_subject = RealSubject()
    client_code(real_subject)

    print('\nClient: Executing the same client code with a proxy:')
    proxy_subject = Proxy(real_subject)
    client_code(proxy_subject)
