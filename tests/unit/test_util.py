from yass import util


def add(a, b, c):
    return a + b + c


class Object(object):

    def add(self, a, b, c):
        return a + b + c


obj = Object()
expected = dict(a=0, b=1, c=2)

# TODO: test this case
# util.map_parameters_in_fn_call([0, 1], dict(b=1), add)


def test_all_kwargs_function():
    assert util.map_parameters_in_fn_call([], dict(a=0, b=1, c=2),
                                          add) == expected


def test_all_args_function():
    assert util.map_parameters_in_fn_call([0, 1, 2], {}, add) == expected


def test_args_kwargs_function():
    assert util.map_parameters_in_fn_call([0], dict(b=1, c=2),
                                          add) == expected


def test_all_kwargs_instance_method():
    assert util.map_parameters_in_fn_call([], dict(a=0, b=1, c=2),
                                          obj.add) == expected


def test_all_args_instance_method():
    assert util.map_parameters_in_fn_call([0, 1, 2], {}, obj.add) == expected


def test_args_kwargs_instance_method():
    assert util.map_parameters_in_fn_call([0], dict(b=1, c=2),
                                          obj.add) == expected
