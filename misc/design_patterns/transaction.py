# Guidelines:
# https://refactoring.guru/ru/design-patterns/command
# https://refactoring.guru/ru/design-patterns/command/python/example

from __future__ import absolute_import
from abc import ABC, abstractmethod


class Command(ABC):
    """
    The Command interface declares a method for executing commands.
    """

    @abstractmethod
    def execute(self) -> None:
        pass


class SimpleCommand(Command):
    """
    SimpleCommand is capable of performing simple operations on their own
    """

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> None:
        print(f"SimpleCommand: {self._payload}")


class Receiver:
    """
    Receiver class contain some important business logic. They know how to perform
    all types of operations associated with executing a request. In fact, any class
    can act as a Receiver
    """

    def operation_1(self, a: str) -> None:
        print('Operation 1: ', a)

    def operation_2(self, b: str) -> None:
        print('Opeation 2: ', b)


class ComplexCommand(Command):
    """
    ComplexCommand delegates more complex operations to others objects called "recipients".
    """

    def __init__(self, receiver: Receiver, a: str, b:str ) -> None:
        """
        ComplexCommand can take one or more recipient objects along with any context data
        via the constructor
        """

        self._receiver = receiver
        self._a = a
        self._b = b

    def execute(self) -> None:
        """
        Commands can delegate execution to Recepient
        :return:
        """
        print('Complex Command: Sending to Receiver...')

        self._receiver.operation_1(self._a)
        self._receiver.operation_2(self._b)


class Invoker:
    """
    The Invoker is associated with one or more Commands. He sends a request to Command
    """

    _on_start = None
    _on_finish = None

    _command = None

    def set_on_start(self, command: Command):
        self._on_start = command

    def set_on_finish(self, command: Command):
        self._on_finish = command

    def execute(self) -> None:
        """
        Invoker is independent of specific command classes and recipients.
        Invoker passes the request to the Recipient indirectly by executing the Command
        """

        if isinstance(self._on_start, Command):
            self._on_start.execute()

        if isinstance(self._on_finish, Command):
            self._on_finish.execute()


if __name__ == '__main__':
    invoker = Invoker()

    # Executing Simple Command
    invoker.set_on_start(SimpleCommand('Hello! I am SimpleCommand'))

    # Executing Complex Command
    receiver = Receiver()
    invoker.set_on_finish(ComplexCommand(receiver, 'Send Email', 'Save Progress'))

    invoker.execute()

