import pytest

from perceptron.model.abc import Observer, Metric


class TestForABC:

    def test_bad_observer(self):
        class BadConcreteObserver(Observer):
            pass

        with pytest.raises(TypeError):
            x = BadConcreteObserver()

        assert not issubclass(BadConcreteObserver, Observer)

    def test_good_observer(self):
        class GoodConcreteObserver(Observer):
            def update(**kwargs):
                pass

        assert issubclass(GoodConcreteObserver, Observer)
        assert isinstance(GoodConcreteObserver(), Observer)

    def test_bad_metric(self):
        class BadConcreteMetric(Metric):
            def update(**kwargs):
                return super().update()

        with pytest.raises(TypeError):
            x = BadConcreteMetric()

        assert issubclass(BadConcreteMetric, Observer)
        assert not issubclass(BadConcreteMetric, Metric)

    def test_good_metric(self):
        class GoodConcreteMetric(Metric):
            def update(**kwargs):
                return super().update()

            def get_metric():
                pass

        assert issubclass(GoodConcreteMetric, Observer)
        assert isinstance(GoodConcreteMetric(), Observer)
        assert issubclass(GoodConcreteMetric, Metric)
        assert isinstance(GoodConcreteMetric(), Metric)

