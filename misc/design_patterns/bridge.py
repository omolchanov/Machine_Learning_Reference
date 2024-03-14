from abc import abstractmethod


class Abstraction:
    """
    Abstraction establishes an interface for the "control" part of two class hierarchies
    """

    def __init__(self, implementation) -> None:
        self.implementation = implementation

    def operation(self) -> str:
        return self.implementation.operation_implementation()


class Implementation:
    """
    Implementation sets the interface for all implementation classes.
    It does not have to conform to the Abstraction interface.
    """

    @abstractmethod
    def operation_implementation(self) -> str:
        pass


class ConcreteImplementationA(Implementation):

    def operation_implementation(self) -> str:
        return 'ConcreteImplementationA on platform A'


class ConcreteImplementationB(Implementation):

    def operation_implementation(self) -> str:
        return 'ConcreteImplementationA on platform B'


def client_code(abstraction: Abstraction) -> None:
    """
    Except during the initialization phase, when an Abstraction object is associated with a specific Implementation
    object, client code must depend only on the Abstraction class.
    :param abstraction
    """

    print(abstraction.operation())


if __name__ == '__main__':
    # Client code should work with any pre-configured combination of Abstraction and Implementation

    imp_a = ConcreteImplementationA()
    client_code(Abstraction(imp_a))

    imp_b = ConcreteImplementationB()
    client_code(Abstraction(imp_b))
