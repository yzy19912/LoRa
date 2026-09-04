def add(a, b):
    return a + b


case = [(1,2,3), (2,3,5)]


class TestAdd:
    def test_add(self):
        for a, b, expected in case:
            result = add(a,b)
            assert result == expected, (f" result {result}")



