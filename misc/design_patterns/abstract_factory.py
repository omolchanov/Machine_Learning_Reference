# Guideline:
# refactoring.guru/ru/design-patterns/abstract-factory
# https://refactoring.guru/ru/design-patterns/abstract-factory/python/example

from __future__ import annotations
from abc import ABC, abstractmethod


class AbstractFactory(ABC):
    """
    Interface for Abstract factory responsible for creating different Products
    """

    @abstractmethod
    def create_product_a(self) -> AbstractProductA:
        pass

    @abstractmethod
    def create_product_b(self) -> AbstractProductB:
        pass


class ConcreteFactory1(AbstractFactory):
    """
    The factory methods responsible for creating particular Products of the same Line
    """

    def create_product_a(self) -> AbstractProductA:
        return ConcreteProductA1()

    def create_product_b(self) -> AbstractProductB:
        return ConcreteProductB1()


class ConcreteFactory2(AbstractFactory):
    """
    The factory methods responsible for creating particular Products of the same Line
    """

    def create_product_a(self) -> AbstractProductA:
        return ConcreteProductA2()

    def create_product_b(self) -> AbstractProductB:
        return ConcreteProductB2()


class AbstractProductA(ABC):
    """
    Interface for Product A
    """

    @abstractmethod
    def get_name_a(self) -> str:
        pass


class ConcreteProductA1(AbstractProductA):
    """
    Responsible for managing Product A of Line 1
    """

    def get_name_a(self) -> str:
        return 'ConcreteProductA1'


class ConcreteProductA2(AbstractProductA):
    """
    Responsible for managing Product A of Line 2
    """

    def get_name_a(self) -> str:
        return 'ConcreteProductA2'


class AbstractProductB(ABC):
    """
    Interface for Product B
    """

    @abstractmethod
    def get_name_b(self) -> str:
        pass

    @abstractmethod
    def collaborate_with(self, collaborator: AbstractProductA) -> None:
        pass


class ConcreteProductB1(AbstractProductB):
    """
    Responsible for managing Product B of Line 1
    """

    def get_name_b(self) -> str:
        return 'ConcreteProductB1'

    def collaborate_with(self, collaborator: AbstractProductA) -> str:
        print(self.get_name_b(), 'collaborating with ', collaborator.get_name_a())


class ConcreteProductB2(AbstractProductB):
    """
    Responsible for managing Product B of Line 2
    """

    def get_name_b(self) -> str:
        return 'ConcreteProductB2'

    def collaborate_with(self, collaborator: AbstractProductA) -> None:
        print(self.get_name_b(), 'collaborating with ', collaborator.get_name_a())


def client_code(factory: AbstractFactory) -> tuple[AbstractProductA, AbstractProductB]:
    """
    Client's code responsible for creating Products of the particular Line
    :param factory: Abstract factory (Line)
    :return: Products of the same Line
    """

    product_a = factory.create_product_a()
    product_b = factory.create_product_b()

    product_b.collaborate_with(product_a)

    return product_a, product_b


if __name__ == '__main__':
    print('Client: Testing client code with the first factory type')
    pr_a, pr_b = client_code(ConcreteFactory1())
    print(pr_a, pr_b)
    print(pr_a.get_name_a())
    print(pr_b.get_name_b())

    print('\nClient: Testing client code with the second factory type')
    pr_a, pr_b = client_code(ConcreteFactory2())
    print(pr_a, pr_b)
    print(pr_a.get_name_a())
    print(pr_b.get_name_b())
