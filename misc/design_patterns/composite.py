# Guideline:
# https://refactoring.guru/ru/design-patterns/composite
# https://refactoring.guru/ru/design-patterns/composite/python/example

from __future__ import annotations
from abc import ABC, abstractmethod


class Component(ABC):
    """
    The base class Component declares common operations for both simple and
    complex structure objects.
    """

    def __init__(self) -> None:
        """
        The base Component can declare an interface for setting and getting
        the parent of a component in a tree structure.
        """
        self._parent = None

    @property
    def parent(self) -> Component:
        return self._parent

    @parent.setter
    def parent(self, parent: Component):
        self._parent = parent

    """
    It is useful to define child control operations right in the base Component class
    in some cases. This way you won't need expose concrete component classes to client code, 
    even while assembling a tree of objects. The disadvantage of this approach is that 
    these methods will be empty for Leaf components.
    """
    def add(self, component: Component) -> None:
        pass

    def remove(self, component: Component) -> None:
        pass

    def is_composite(self) -> bool:
        """
        A method that allows client code to understand whether a component
        can have nested objects.
        """

        return False

    @abstractmethod
    def get_price(self) -> float:
        """
        The Base Component may itself implement some default behaviorr delegate this to
        concrete classes.
        """
        pass


class Leaf(Component):
    """
    The Leaf class represents the final object of a structure. Leaf can't have nested
    components.
    """

    def get_price(self) -> float:
        return 24.5


class Composite(Component):
    """
    The Composite class contains complex components that can have nested Components.
    Typically, Container objects delegate the actual work to their children,
    and then “sum up” the result.
    """

    def __init__(self) -> None:
        super().__init__()
        self._children = []

    def add(self, component: Component) -> None:
        self._children.append(component)
        component.parent = self

    def remove(self, component: Component) -> None:
        self._children.remove(component)
        component.parent = None

    def is_composite(self) -> bool:
        return True

    def get_price(self) -> float:
        """
        Container performs its core logic in a specific way. He's walking recursively
        through all its children, collecting and summing up their results.
        """

        prices = []

        for child in self._children:
            prices.append(child.get_price())

        return sum(prices)


def client_code(component: Component) -> None:
    """
    The client code interacts with all Components through a basic interface.
    """

    print('Price: ', component.get_price())


def client_code2(component1: Component, component2: Component) -> None:
    """
    The client code can work with both simple and complex Components,
    regardless of their specific classes.
    """

    if component1.is_composite():
        component1.add(component2)

    print('Price: ', component1.get_price())


if __name__ == '__main__':

    # Working with simple Component
    simple = Leaf()

    print('Client code: got a simple Component')
    client_code(simple)

    branch1 = Composite()
    branch1.add(Leaf())
    branch1.add(Leaf())

    print('Client code: got a composite Component')
    client_code(branch1)

    print('Client code: got a simple and a composite Component')
    client_code2(branch1, simple)

    branch2 = Composite()
    branch2.add(Leaf())

    tree = Composite()
    tree.add(branch1)
    tree.add(branch2)

    print('Client code: got a tree Component')
    client_code(tree)
    client_code2(tree, simple)
