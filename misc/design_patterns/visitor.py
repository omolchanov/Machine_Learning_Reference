# Guidelines:
# https://refactoring.guru/ru/design-patterns/visitor
# https://refactoring.guru/ru/design-patterns/visitor/python/example

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Component(ABC):
    """
    The Component interface declares an accept method the accepts Visitor object
    """

    @abstractmethod
    def accept(self, visitor: Visitor) -> None:
        pass


class ConcreteComponentA(Component):
    """
    Each Concrete Component must implement the accept method in a way,
    so that it calls the Visitor method corresponding to the component class
    """

    def accept(self, visitor: Visitor) -> None:
        """
        Note that we are calling visitComponentA, what matches the name of the current class.
        This way Visitor is able to find out which component class it is working with
        """

        visitor.visit_component_a(self)

    def method1(self) -> str:
        """
        Concrete Components may have special methods not declared in their base class or interface.
        Visitor can use these methods because it knows about the concrete class of the Component
        """

        return 'method1 of Component A'


class ConcreteComponentB(Component):

    def accept(self, visitor: Visitor) -> None:
        visitor.visit_component_b(self)

    def method2(self) -> str:
        return 'method2 of Component B'


class Visitor(ABC):
    """
    The Visitor interface declares a set of visiting methods corresponding to Component classes.
    The visiting method signature allows the visitor defines the specific Component class it deals with.
    """

    @abstractmethod
    def visit_component_a(self, element: ConcreteComponentA) -> None:
        pass

    @abstractmethod
    def visit_component_b(self, element: ConcreteComponentB) -> None:
        pass


"""
Concrete Visitors implement multiple versions of the same algorithm,
that can work with all classes of concrete Components.
"""


class ConcreteVisitor1(Visitor):

    def visit_component_a(self, element: ConcreteComponentA) -> None:
        print(f"{element.method1()} + ConcreteVisitor1")

    def visit_component_b(self, element: ConcreteComponentB) -> None:
        print(f"{element.method2()} + ConcreteVisitor1")


class ConcreteVisitor2(Visitor):

    def visit_component_a(self, element: ConcreteComponentA) -> None:
        print(f"{element.method1()} + ConcreteVisitor2")

    def visit_component_b(self, element: ConcreteComponentB) -> None:
        print(f"{element.method2()} + ConcreteVisitor2")


def client_code(components: List[Component], visitor: Visitor) -> None:
    for component in components:
        component.accept(visitor)


if __name__ == '__main__':
    # Client code can perform Visitor's operations on any set of elements without finding out
    # their specific classes

    components = [
        ConcreteComponentA(),
        ConcreteComponentB()
    ]

    print('Visitor 1:')
    client_code(components, ConcreteVisitor1())

    print('\nVisitor 2:')
    client_code(components, ConcreteVisitor2())
