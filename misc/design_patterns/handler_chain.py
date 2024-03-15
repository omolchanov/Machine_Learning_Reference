# Guideline:
# https://refactoring.guru/ru/design-patterns/chain-of-responsibility/python/example
# https://refactoring.guru/ru/design-patterns/chain-of-responsibility

from abc import ABC, abstractmethod
from typing import Any, Optional


class Handler(ABC):
    """
    The Handler interface declares a method for constructing a chain of handlers.
    It also declares a method for executing the request.
    """

    @abstractmethod
    def set_next(self, hanlder):
        pass

    @abstractmethod
    def handle(self, request) -> Optional[str]:
        pass


class AbstractHandler(Handler):
    """
    Default chaining behavior can be implemented within a base class Handler.
    """

    _next_handler: Handler = None

    def set_next(self, hanlder: Handler) -> Handler:
        self._next_handler: Handler = hanlder

        return hanlder

    @abstractmethod
    def handle(self, request: Any) -> str:
        if self._next_handler:
            return self._next_handler.handle(request)

        return None

# The concrete Handlers either process the request or forward it to the next Handler in the chain.


class MonkeyHandler(AbstractHandler):

    def handle(self, request: Any) -> str:
        if request == 'Banana':
            return f'Monkey: I will eat the {request}'

        else:
            return super().handle(request)


class SquirellHandler(AbstractHandler):

    def handle(self, request: Any) -> str:
        if request == 'Nut':
            return f'Squirell: I will eat the {request}'

        else:
            return super().handle(request)


class DogHandler(AbstractHandler):

    def handle(self, request: Any) -> str:
        if request == 'Meatball':
            return f'Dog: I will eat the {request}'

        else:
            return super().handle(request)


def client_code(handler: Handler):
    for food in ['Nut', 'Cup of coffee', 'Banana', 'Meatball']:
        print(f'\nWho wants a {food} ?')

        result = handler.handle(food)
        if result:
            print(f'  {result}')
        else:
            print(f'  {food} was left untouched.')


if __name__ == '__main__':
    monkey = MonkeyHandler()
    squirell = SquirellHandler()
    dog = DogHandler()

    monkey.set_next(squirell).set_next(dog)

    # The client should be able to send a request to any Handler, not only to the first one in the chain
    print('Chain: Monkey > Squirrel > Dog')
    client_code(monkey)
    print('================')

    print('Subchain: Squirrel > Dog')
    client_code(squirell)
