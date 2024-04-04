# Gudidelines:
# https://refactoring.guru/ru/design-patterns/iterator
# https://refactoring.guru/ru/design-patterns/iterator/python/example

from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Any

"""
There are two abstract classes from the built-in collections module in Pyhton for building an iterator - 
Iterable, Iterator. There is a need to implement the __iter__() method in iterable object (list), 
and the __next__() method in the iterator.
"""


class AlphabeticalOrderIterator(Iterator):
    """
    Specific Iterators implement different traversal algorithms.
    Such classes permanently store the current bypass position.
    """

    # The _position attribute stores the current position of the traversal
    _position: int = None

    # This attribute specifies the direction of traversal
    _reverse: bool = False

    def __init__(self, collection: WordsCollection, reverse: bool = False) -> None:
        self._collection = collection
        self._reverse = reverse
        self._position = -1 if reverse else 0

    def __next__(self) -> Any:
        """
        The __next__() method returns the next element in the sequence.
        When reaching the end of the collection and in subsequent calls,
        the StopIteration exception should be called
        """

        try:
            value = self._collection[self._position]
            self._position += -1 if self._reverse else 1

        except IndexError:
            raise StopIteration()

        return value


class WordsCollection(Iterable):
    """
    Concrete Collections provide one or more methods for obtaining new instances
    of Iterator compatible with the Collection class
    """

    def __init__(self, collection: list[Any] | None = None) -> None:
        self._collection = collection or []

    def __getitem__(self, index: int) -> Any:
        return self._collection[index]

    def __iter__(self) -> AlphabeticalOrderIterator:
        """
        The __iter__() method returns an Iterator object.
        Iterator is sorted in ascending order by default.
        """

        return AlphabeticalOrderIterator(self)

    def get_reverse_iterator(self) -> AlphabeticalOrderIterator:
        return AlphabeticalOrderIterator(self, True)

    def add_item(self, item: Any) -> None:
        self._collection.append(item)


if __name__ == '__main__':

    # Client code may or may not know about the Concrete Iterator or Collections
    collection = WordsCollection()

    collection.add_item('First')
    collection.add_item('Second')
    collection.add_item('Third')

    print('Straight traversal:')
    print('\n'.join(collection))

    print('\nReverse traversal:')
    print('\n'.join(collection.get_reverse_iterator()))
