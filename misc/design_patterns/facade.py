# Guideline:
# https://refactoring.guru/ru/design-patterns/facade
# https://refactoring.guru/ru/design-patterns/facade/python/example

from __future__ import annotations


class Facade:
    """
    The Facade  provides a simple interface for complex logic of one or more subsystems.
    The Facade delegates customer requests to the appropriate objects within the subsystem.
    The Facade is also responsible for managing their lifecycle.
    """

    def __init__(self, ss1: Subsystem1, ss2: Subsystem2) -> None:
        """
        Depending on the application's needs, you can provide Facade with existing subsystem
        objects  or force Facade to create them.
        """

        self.ss1 = ss1 or Subsystem1()
        self.ss2 = ss2 or Subsystem2()

    def operation(self) -> None:

        print('Facade inits the both subsystems:')
        print(self.ss1.init_ss())
        print(self.ss2.init_ss())

        print('\nFacade orders subsystems to perform the action:')
        print(self.ss1.aggregate_text())
        print(self.ss2.compress_images())


class Subsystem1:
    """
    The Subsystem can accept requests either from the Facade or from the Client directly.
    Facade is another client, and it is not part of the Subsystem.
    """

    def init_ss(self) -> str:
        return 'SS1 is ready'

    def aggregate_text(self) -> str:
        return 'Aggregating text...'


class Subsystem2:
    def init_ss(self) -> str:
        return 'SS2 is on!'

    def compress_images(self) -> str:
        return 'Compressing images'


def client_code(facade: Facade) -> None:
    """
    The Client code works with complex subsystems through a simple interface, provided by Facade.
    When a Facade manages the lifecycle of a Subsystem, the Client may not even be aware of the Subsystem.
    """
    facade.operation()


if __name__ == '__main__':
    # Some Subsystem objects may already be created in the Client code.
    # In this case it may be useful to initialize the Facade with these
    # objects instead of allowing the Facade to create new instances.

    f = Facade(ss1=None, ss2=None)
    client_code(f)
