import pytest
@pytest.fixture(scope="session")

# test_calculator.py

def test_addition():
    assert 2 + 2 == 4

def test_subtraction():
    assert 5 - 3 == 2

def test_multiplication():
    assert 3 * 4 == 12

def test_division():
    assert 10 / 2 == 5

def test_modulo():
    assert 10 % 3 == 1

def test_exponentiation():
    assert 2 ** 3 == 8

def test_floor_division():
    assert 10 // 3 == 3

def test_string_concatenation():
    assert "hello" + " " + "world" == "hello world"

def test_list_concatenation():
    assert [1, 2] + [3, 4] == [1, 2, 3, 4]

def test_set_intersection():
    assert set([1, 2, 3]) & set([2, 3, 4]) == set([2, 3])

def test_dictionary_lookup():
    d = {"a": 1, "b": 2, "c": 3}
    assert d["b"] == 2

def test_tuple_unpacking():
    a, b, c = (1, 2, 3)
    assert a == 1 and b == 2 and c == 3

if __name__ == '__main__':
    pytest.main()
