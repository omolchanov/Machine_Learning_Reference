# Guideline:
# https://refactoring.guru/ru/design-patterns/builder
# https://refactoring.guru/ru/design-patterns/builder/python/example

from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    """
     The Builder interface declares creation methods for various parts of Products
    """

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def produce_part_a(self) -> None:
        pass

    @abstractmethod
    def produce_part_b(self) -> None:
        pass

    @abstractmethod
    def produce_part_c(self) -> None:
        pass


class ConcreteBuilder1(Builder):
    """
    Concrete Builder classes implements the Builder interface and provide
    the specific implementations of construction steps.
    """

    def __init__(self) -> None:
        self._product = None

        # New builder instance must contain an empty Product object
        # which is used for future constructions
        self.reset()

    def reset(self) -> None:
        self._product = Product1()

    @property
    def product(self) -> None:
        """
        Concrete Builders must provide their own methods. This is because different types of Builders
        can create completely different Products with different interfaces.
        :return: Product
        """

        product = self._product

        # Typically, once the final result was returned to the Client, the Builder is ready
        # to start producing the next Product.
        # Therefore, it is common practice to call the reset method at the end of the body
        # getProduct method
        self.reset()

        return product

    def produce_part_a(self) -> None:
        self._product.add('PartA1')

    def produce_part_b(self) -> None:
        self._product.add('PartB1')

    def produce_part_c(self) -> None:
        self._product.add('PartC1')


class Product1:
    """
    Results produced by different Builders may not always implement the same interface.
    """

    def __init__(self) -> None:
        self.parts = []

    def add(self, part: Any) -> None:
        self.parts.append(part)

    def list_parts(self) -> None:
        print(self.parts)


class Director:
    """
    Director is only responsible for executing the construction steps in a specific order.
    This is useful when creating Products in a certain order or special configuration.
    The Director class is optional, since the Client can directly manage the Builder.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder):
        self._builder = builder

    # Director can build Products following numerous variations using the same construction steps
    def build_minimal_product(self) -> None:
        self._builder.produce_part_a()

    def build_full_product(self) -> None:
        self._builder.produce_part_a()
        self._builder.produce_part_b()
        self._builder.produce_part_c()


if __name__ == '__main__':
    # The Client's code creates a Builder object, passes it to the Director, and then initiates
    # the construction process. The final result is extracted from the Builder

    # Running the Builder with Director
    print('Running the Builder with Director')

    director = Director()
    bld_1 = ConcreteBuilder1()

    director.builder = bld_1

    print('Constructing minimal Product')
    director.build_minimal_product()
    bld_1.product.list_parts()

    print('\nConstructing full Product')
    director.build_full_product()
    bld_1.product.list_parts()

    print('==============')

    # Running the Builder without Director
    print('\nRunning the Builder only')

    bld_2 = ConcreteBuilder1()

    bld_2.produce_part_a()
    bld_2.produce_part_b()

    bld_2.product.list_parts()
