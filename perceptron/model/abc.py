from abc import ABC, abstractmethod


def assert_callable_attr(cls, attr):
    return hasattr(cls, attr) and callable(cls.__dict__.get(attr))


class Observer(ABC):
    """An Observer interface to allow subclasses to be used in the model.

    All subclasses of `Observer`, wether real (using regular inheritance) or
    virtual (using `Observer.register`), are required to implement all the
    abstract methods defined in `Observer`. The `issubclass(Foo, Observer)` check
    will only succeed if `Foo` implements all the abstract methods.
    """

    @abstractmethod
    def update(**kwargs):
        raise NotImplemented()

    @classmethod
    def __subclasshook__(cls, subclass):
        return assert_callable_attr(subclass, 'update')


class Metric(Observer):
    """A specialized `Observer` interface that allows for numerical metrics to be
    calculated from the model's parameters and used in the model.
    """

    @abstractmethod
    def update(**kwargs):
        return super().update()

    @abstractmethod
    def get_metric():
        pass

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            super().__subclasshook__(subclass) and
            assert_callable_attr(subclass, 'get_metric')
        )


