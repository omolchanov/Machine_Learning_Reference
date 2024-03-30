# Guideline:
# https://refactoring.guru/ru/design-patterns/decorator
# https://refactoring.guru/ru/design-patterns/decorator/python/example

class Component:
    def operation(self) -> str:
        pass


class ConcreteComponent(Component):
    def operation(self) -> str:
        return 'Concrete Component'


class Decorator(Component):

    _component: Component = None

    def __init__(self, component: Component):
        self._component: Component = component

    @property
    def component(self) -> Component:
        return self._component

    def operation(self) -> str:
        return self._component.operation()


class ConcreteDecoratorA(Decorator):

    def operation(self) -> str:
        return f"ConcreteDecoratorA: ({self.component.operation()})"


class ConcreteDecoratorB(Decorator):

    def operation(self) -> str:
        return f"ConcreteDecoratorB: ({self.component.operation()})"


def client_code(component: ConcreteComponent) -> None:
    print(f"RESULT: {component.operation()}")


if __name__ == '__main__':

    # Simple Component
    print('Client: got a simple Component')

    simple_comp = ConcreteComponent()
    client_code(simple_comp)

    # Decorated Component
    print('\nClient: got a decorated component')
    decorator1 = ConcreteDecoratorA(simple_comp)
    client_code(decorator1)

    # Decorated Decorator
    print('\nClient: got a decorated Decorator')
    decorator2 = ConcreteDecoratorB(decorator1)
    client_code(decorator2)
