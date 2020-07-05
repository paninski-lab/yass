from yass.batch import vectorize_parameter


# create a function that adds up three numbers, vectorize over parameter a
@vectorize_parameter('a')
def add(a, b, c):
    return a + b + c


# same as above, but this time decorate an instance method
class Object(object):
    @vectorize_parameter('a')
    def add(self, a, b, c):
        return a + b + c


add([0, 1, 2, 3, 4], 1, 0)

o = Object()
o.add([0, 1, 2, 3, 4], 1, 0)
