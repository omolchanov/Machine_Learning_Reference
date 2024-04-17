# Guidelines:
# https://refactoring.guru/ru/design-patterns/template-method
# https://refactoring.guru/ru/design-patterns/template-method/python/example

from abc import ABC, abstractmethod


class AbstractClass(ABC):
    """
    Abstract Class defines a template method containing the skeleton of some algorithms
    recalling abstract primitive methods.

    Concrete Classes must implement these operations, but leave the template method itself unchanged.
    """

    def template_method(self):
        self.base_op1()
        self.required_op2()
        self.base_op2()
        self.hook1()
        self.required_op2()
        self.base_op3()
        self.hook2()

    """
    These operations have implementations
    """

    def base_op1(self) -> None:
        print('Abstract Class: base_op1')

    def base_op2(self) -> None:
        print('Abstract Class: base_op2')

    def base_op3(self) -> None:
        print('Abstract Class: base_op3')

    """
    These operations must be implemented in subclasses
    """

    @abstractmethod
    def required_op1(self) -> None:
        pass

    @abstractmethod
    def required_op2(self) -> None:
        pass

    """
    These are "hooks". Concrete Classes can override them, but this is not required.
    Hooks have a standard (but empty) implementation. 
    Hooks provide additional extension points at some critical places in the algorithm.
    """
    def hook1(self) -> None:
        pass

    def hook2(self) -> None:
        pass


class ConcreteClass1(AbstractClass):
    """
    Concrete classes must implement all the abstract operations of the Abstract class.
    They can also override some operations with the default implementation
    """

    def required_op1(self) -> None:
        print('Concrete Class 1: req_operation1')

    def required_op2(self) -> None:
        print('Concrete Class 1: req_operation2')


class ConcreteClass2(AbstractClass):
    """
    Usually Concrete classes override only part of the operations of the Abstract class
    """
    def required_op1(self) -> None:
        print('Concrete Class 2: req_operation1')

    def required_op2(self) -> None:
        print('Concrete Class 2: req_operation2')

    def hook1(self) -> None:
        print('Concrete Class 2: overriding hook1')


def client_code(abstract_class: AbstractClass) -> None:
    """
    Client code calls a template method to execute the algorithm.
    Client code is not familiar with the Concrete class it is working with,
    as long as it is working with objects through their base class interface.
    """

    abstract_class.template_method()


if __name__ == '__main__':
    print('Same client code can work with different subclasses', ConcreteClass1())
    client_code(ConcreteClass1())

    print('\nSame client code can work with different subclasses', ConcreteClass2())
    client_code(ConcreteClass2())
