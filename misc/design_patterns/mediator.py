# Guidelines:
# https://refactoring.guru/ru/design-patterns/mediator
# https://refactoring.guru/ru/design-patterns/mediator/python/example

from __future__ import annotations
from abc import ABC


class Mediator(ABC):
    """
    The Mediator interface provides a method that the components use to notify
    the Mediator about various events. The Mediator can respond to such events
    and pass execution to other components.
    """

    def notify(self, sender: object, event: str) -> None:
        pass


class ConcreteMediator(Mediator):

    def __init__(self, component1: Component1, component2: Component2) -> None:
        self._component1 = component1
        self._component1.mediator = self

        self._component2 = component2
        self._component2.mediator = self

    def notify(self, sender: object, event: str) -> None:
        if event == 'A':
            print('Mediator reacts on Event A. Triggering operations:')
            self._component2.do_c()

        if event == 'D':
            print('Mediator reacts on Event D. Triggering operations:')
            self._component1.do_b()
            self._component2.do_c()


class BaseComponent:
    """
    The Base Component provides basic functionality of storing
    the Mediator withing the Component object
    """

    def __init__(self, mediator: Mediator = None) -> None:
        self._mediator = mediator

    @property
    def mediator(self) -> Mediator:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator


"""
Specific Components implement different functionality and do not depend on other Components. 
They also do not depend on any specific intermediary classes
"""


class Component1(BaseComponent):

    def do_a(self) -> None:
        print('Component 1 does A')
        self.mediator.notify(self, 'A')

    def do_b(self) -> None:
        print('Component 1 does B')
        self.mediator.notify(self, 'B')


class Component2(BaseComponent):

    def do_c(self) -> None:
        print('Component 2 does C')
        self.mediator.notify(self, 'C')

    def do_d(self) -> None:
        print('Component 2 does D')
        self.mediator.notify(self, 'D')


if __name__ == '__main__':

    # Client code
    c1 = Component1()
    c2 = Component2()

    m = ConcreteMediator(c1, c2)

    print('Client triggers operation A')
    c1.do_a()

    print('\nClient triggers operation D')
    c2.do_d()
