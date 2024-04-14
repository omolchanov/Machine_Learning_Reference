# Guidelines:
# https://refactoring.guru/ru/design-patterns/state
# https://refactoring.guru/ru/design-patterns/state/python/example

from __future__ import annotations
from abc import ABC, abstractmethod


class Context:
    """
    Context defines the interface of interest to clients. It also stores a reference to
    the State object proividing current Context state
    """

    _state = None  # A reference to the current state of the Context

    def __init__(self, state: State) -> None:
        self.transition_to(state)

    def transition_to(self, state: State) -> None:
        """
        Context allows changing the State object during the runtime
        """

        print('Transitioning to ', type(state).__name__)
        self._state = state
        self._state.context = self

    """
    Context delegates some of its behavior to the current State object
    """

    def request1(self):
        self._state.handle1()

    def request2(self):
        self._state.handle2()


class State(ABC):
    """
    The base State class declares methods that the Specific States must implement
    and also provides a back reference to the Context object associated with the State.
    This backlink can be used for transfering Context to another State
    """

    def __init__(self):
        self._context = None

    @property
    def context(self) -> Context:
        return self._context

    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @abstractmethod
    def handle1(self) -> None:
        pass

    @abstractmethod
    def handle2(self) -> None:
        pass


"""
Specific States implement different behavior patterns associated with State of the Context
"""


class ConcreteStateA(State):

    def handle1(self) -> None:
        print('ConcreteStateA handles request1')
        print('ConcreteStateA wants to change the state of the context')
        self.context.transition_to(ConcreteStateB())

    def handle2(self) -> None:
        print('ConcreteStateA handles request1')


class ConcreteStateB(State):

    def handle1(self) -> None:
        print('ConcreteStateB handles request1')

    def handle2(self) -> None:
        print('ConcreteStateB handles request2')
        print('ConcreteStateB wants to change the state of the context')
        self.context.transition_to(ConcreteStateA())


if __name__ == '__main__':

    # The client code
    c = Context(ConcreteStateA())

    c.request1()
    c.request2()
