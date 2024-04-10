# Guidelines:
# https://refactoring.guru/ru/design-patterns/observer
# https://refactoring.guru/ru/design-patterns/observer/python/example

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List


class Publisher(ABC):
    """
    The Publisher interface declares a set of methods for managing Listeners
    """

    def __init__(self):
        self._state = None

    @abstractmethod
    def attach(self, listener: Listener) -> None:
        pass

    @abstractmethod
    def detach(self, listener: Listener) -> None:
        pass

    @abstractmethod
    def notify(self) -> None:
        """
        Notifies all Listeners about the event.
        """
        pass

    @property
    def state(self):
        return self._state


class ConcretePublisher(Publisher):
    """
    The Concrete Publisher owns state and notifies Listeners when it has changed
    """

    # Stores the state of the Publisher required for Listeners
    _state: int = None

    # List of Listeners. It also could be stored in more detail (classified by event type, etc.)
    _listeners: List[Listener] = []

    def attach(self, listener: Listener) -> None:
        print('\nSubject. Attached a Listener', listener)
        self._listeners.append(listener)

    def detach(self, listener: Listener) -> None:
        print('\nSubject. Detached a Listener', listener)
        self._listeners.remove(listener)

    def notify(self) -> None:
        """
        Running an update in each Listener
        """

        print('\nNotifying Listeners:')
        for ln in self._listeners:
            ln.update(self)

    """
    Typically the subscription logic is only part of what the Publisher does.
    Publishers often contain some important business logic that runs a notification 
    method whenever something important is about to happen (or after that).
    """

    def generate(self) -> None:
        self._state = random.randrange(0, 10)
        print('\nPublisher. Updating state to ', self._state)

        self.notify()


class Listener(ABC):
    """
    The Listener interface declares a notification method that Publishers
    use to notify their subscribers.
    """

    @abstractmethod
    def update(self, publisher: Publisher):
        pass


"""
Concrete Listeners react to updates released by the Publisher they are attached to.
"""


class ConcreteListenerA(Listener):

    def update(self, publisher: Publisher) -> None:
        if publisher.state > 5:
            print('ConcreteListenerA reacted to the event')


class ConcreteListenerB(Listener):

    def update(self, publisher: Publisher) -> None:
        if publisher.state == 0:
            print('ConcreteListenerB reacted to the event')


if __name__ == '__main__':
    
    # Client code
    p = ConcretePublisher()

    listener_a = ConcreteListenerA()
    p.attach(listener_a)

    listener_b = ConcreteListenerB()
    p.attach(listener_b)

    p.generate()
    p.generate()

    p.detach(listener_a)

    p.generate()
