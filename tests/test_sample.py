import pytest

def add(a: float, b: float) -> float:
    """返回两个数的和"""
    return a + b

def subtract(a: float, b: float) -> float:
    """返回两个数的差"""
    return a - b

def multiply(a: float, b: float) -> float:
    """返回两个数的积"""
    return a * b

def divide(a: float, b: float) -> float:
    """返回两个数的商，除数为0时抛出ValueError"""
    if b == 0:
        raise ValueError("除数不能为0")
    return a / b

def test_add():
    """测试加法功能"""
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(2.5, 3.5) == 6.0

def test_subtract():
    """测试减法功能"""
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2
    assert subtract(0, 0) == 0
    assert subtract(4.5, 2.5) == 2.0

def test_multiply():
    """测试乘法功能"""
    assert multiply(2, 3) == 6
    assert multiply(-1, 1) == -1
    assert multiply(0, 5) == 0
    assert multiply(2.5, 4) == 10.0

def test_divide():
    """测试除法功能"""
    assert divide(6, 3) == 2
    assert divide(-6, 3) == -2
    assert divide(5, 2) == 2.5

    # 测试异常情况
    with pytest.raises(ValueError, match="除数不能为0"):
        divide(5, 0)

# 参数化测试示例
@pytest.mark.parametrize("a, b, expected", [
    (2, 3, 5),
    (-1, 1, 0),
    (0, 0, 0),
    (2.5, 3.5, 6.0),
])

def test_add_parametrized(a, b, expected):
    """参数化测试加法功能"""
    assert add(a, b) == expected
