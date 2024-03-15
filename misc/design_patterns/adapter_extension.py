# Guideline:
# https://refactoring.guru/ru/design-patterns/adapter
# https://refactoring.guru/ru/design-patterns/adapter/python/example


class Target:
    """
    The Target class declares an interface the Client code can interact with
    """

    def request(self) -> str:
        return 'Target: The default targets behavior'


class Adaptee:
    """
    The Adaptee class contains some useful behavior, but its
    interface is incompatible with the Client code
    """

    def specific_request(self) -> str:
        return 'eetpadA eht fo roivaheb laicepS'


class Adapter(Target, Adaptee):
    """
    Adapter makes the interface of Adaptee class compatible with Target interface
    with the multiple inheritance.
    """

    def request(self) -> str:
        return f'Adapter (TRANSLATED) {self.specific_request()[::-1]}'


def client_code(target: Target) -> None:
    """
    Client code supports all classes that use the Target interface
    :param target: Target
    """

    print(target.request())


if __name__ == '__main__':
    print('Client: I can work just fine with the Target objects:')
    target = Target()
    client_code(target)

    adaptee = Adaptee()
    print(f'\nClient: The Adaptee class has a weird interface. {adaptee.specific_request()}')

    adapter = Adapter()
    print("\nClient: But I can work with it via the Adapter:")
    client_code(adapter)
