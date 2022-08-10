def get_concrete_class(AbstractClass, *args):
    """Create concrete child of abstract class for unit testing"""

    class ConcreteClass(AbstractClass):
        def __init__(self, *args) -> None:
            super().__init__(*args)

    ConcreteClass.__abstractmethods__ = frozenset()
    return type("DummyConcreteClassOf" + AbstractClass.__name__, (ConcreteClass,), {})
