# Guidelines:
# https://refactoring.guru/ru/design-patterns/prototype/python/example
# https://refactoring.guru/ru/design-patterns/prototype

import copy


class SelfReferencingEntity:
    def __init__(self):
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class SomeClonableComponent:
    """
    Python provides its own interface of Prototype via `copy.copy` and
    `copy.deepcopy` functions. And any class that wants to implement custom
    implementations have to override `__copy__` and `__deepcopy__` member
    functions.
    """

    def __init__(self, some_int: int, obj_list: list, dep: SelfReferencingEntity):
        self. some_int = some_int
        self.obj_list = obj_list
        self.dep = dep

    def __copy__(self):
        """
        Creates a shallow copy. This method will be called whenever someone calls
        `copy.copy` with this object and the returned value is returned as the
        new shallow copy.

        :return: copy of the Object
        """

        new = self.__class__(self.some_int, self.obj_list, self.dep)
        new.__dict__.update(self.__dict__)

        return new

    def __deepcopy__(self, memo=None):
        """
        Creates a deep copy. This method will be called whenever someone calls
        `copy.deepcopy` with this object and the returned value is returned as
        the new deep copy.

        What is the use of the argument `memo`? Memo is the dictionary that is
        used by the `deepcopy` library to prevent infinite recursive copies in
        instances of circular references. Pass it to all the `deepcopy` calls
        you make in the `__deepcopy__` implementation to prevent infinite
        recursions.
        :param memo: dict
        :return: copy of the Object
        """

        if memo is None:
            memo = {}

        some_int = copy.deepcopy(self.some_int, memo)
        obj_list = copy.deepcopy(self.obj_list, memo)
        dep = copy.deepcopy(self.dep, memo)

        new = self.__class__(self.some_int, self.obj_list, self.dep)
        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


if __name__ == '__main__':

    comp_obj1 = SomeClonableComponent(2, [1, 2, 3], SelfReferencingEntity())

    # Creating a shallow copy of the object
    comp_obj2 = comp_obj1.__copy__()

    print('Original object: ', comp_obj1)
    print('The original object`s properties: ', comp_obj1.__dict__)

    # Changing some properties in the cloned object
    comp_obj2.obj_list.append('test3')
    comp_obj2.some_int = 99

    print('\nCloned object: ', comp_obj2)
    print('The cloned object`s properties: ', comp_obj2.__dict__)

    # Creating a deep copy of the object
    comp_obj3 = comp_obj1.__deepcopy__()

    print('\nDeeply cloned object: ', comp_obj3.__dict__)
