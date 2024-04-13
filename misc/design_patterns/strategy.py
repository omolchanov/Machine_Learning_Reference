# Guidelines
# https://refactoring.guru/ru/design-patterns/strategy
# https://refactoring.guru/ru/design-patterns/strategy/python/example

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Context:
    """
    Context defines the interface of interest to Clients
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Context accepts the Strategy via the constructor.
        Also provides a setter to change during the runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        Context stores a link to the Strategy objects. Context is not familiar with
        specific class of Strategy. It works with all Strategies through the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Context allows replacing the Strategy object during the runtime
        """

        self._strategy = strategy

    def do_something(self) -> None:
        """
        Instead of implementing multiple versions of the algorithm,
        Context delegates the work to the Strategy object.
        """

        print('Context: Sorting data using the strategy ', self._strategy)
        result = self._strategy.do_algorithm([4, 3, 2, 1, 7])

        print(result)


class Strategy(ABC):
    """
    Strategy interface declares common operations for all supported algorithms.
    Context uses this interface to call the algorithm defined for specific Strategy
    """

    @abstractmethod
    def do_algorithm(self, data: List):
        pass


"""
Concrete  Strategies implement the algorithm by following the basic Strategies interface.
This interface makes them interchangeable in Context.
"""


class ConcreteStrategyA(Strategy):
    def do_algorithm(self, data: List) -> List:
        return sorted(data)


class ConcreteStrategyB(Strategy):
    def do_algorithm(self, data: List) -> List:
        return list(reversed(sorted(data)))


if __name__ == '__main__':
    # The client code selects a specific Strategy and passes it to the Context.
    # The client must be aware of the differences between Strategies

    context = Context(ConcreteStrategyA())

    print('Strategy: normal sorting')
    context.do_something()

    print('\nStrategy: reverse sorting')
    context.strategy = ConcreteStrategyB()

    context.do_something()
