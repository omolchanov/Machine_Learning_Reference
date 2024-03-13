# Guideline:
# https://refactoring.guru/ru/design-patterns/factory-method
# https://refactoring.guru/ru/design-patterns/factory-method/python/example

from abc import ABC, abstractmethod


class Creator(ABC):
    """
    Declares a factory method thar returns Product object.
    Might contain some basic business logic that is based on Product objects
    returned by the factory method.
    """

    @abstractmethod
    def get_product(self):
        pass

    def create_product(self) -> str:

        # Calls the factory method to get a Product
        product = self.get_product()

        # Returns a particular instance of Product
        print('Product %s was created' % product.get_name())
        return product


class Creator1(Creator):
    """
    Intends for creating and managing a particular Product (Product1)
    Inherits abstract method from Creator class ['get_product']
    """

    def get_product(self):
        return Product1()


class Creator2(Creator):
    """
    Intends for creating and managing a particular Product (Product2)
    Inherits abstract method from Creator class ['get_product']
    """

    def get_product(self):
        return Product2()


class Product(ABC):
    """
    Interface of Product
    """

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def use(self):
        pass

    @abstractmethod
    def recycle(self):
        pass


class Product1(Product):
    def get_name(self) -> str:
        return 'Product1'

    def use(self):
        return 'Using of Product 1'

    def recycle(self):
        return 'Recycling of Product 1'


class Product2(Product):
    def get_name(self):
        return 'Product2'

    def use(self):
        return 'Using of Product 2'

    def recycle(self):
        return 'Product 2 recyled'


def clinet_code(creator: Creator) -> Product:
    """
    Client's code responsible for cooperating with a particular Creator
    :param creator: a particular Creator
    :return: a Product created by a particular Creator
    """

    return creator.create_product()


if __name__ == '__main__':
    print('App: Launched with the Creator1')
    pr1 = clinet_code(Creator1())
    print(pr1.use())

    print('\nApp: Launched with the Creator2')
    pr2 = clinet_code(Creator2())
    print(pr2.recycle())
