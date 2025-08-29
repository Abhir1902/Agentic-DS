"""
Basic Mathematics Operations Module

This module provides fundamental arithmetic operations including
addition, subtraction, multiplication, and division.

Functions:
    add(a, b): Returns the sum of two numbers
    subtract(a, b): Returns the difference of two numbers  
    multiply(a, b): Returns the product of two numbers
    divide(a, b): Returns the quotient of two numbers
"""

def add(a, b):
    """
    Add two numbers together.
    
    Args:
        a (int, float): First number
        b (int, float): Second number
        
    Returns:
        int, float: Sum of a and b
        
    Example:
        >>> add(5, 3)
        8
        >>> add(2.5, 1.5)
        4.0
    """
    return a + b


def subtract(a, b):
    """
    Subtract second number from first number.
    
    Args:
        a (int, float): First number (minuend)
        b (int, float): Second number (subtrahend)
        
    Returns:
        int, float: Difference of a and b
        
    Example:
        >>> subtract(10, 4)
        6
        >>> subtract(7.5, 2.5)
        5.0
    """
    return a - b


def multiply(a, b):
    """
    Multiply two numbers together.
    
    Args:
        a (int, float): First number
        b (int, float): Second number
        
    Returns:
        int, float: Product of a and b
        
    Example:
        >>> multiply(6, 7)
        42
        >>> multiply(3.5, 2)
        7.0
    """
    return a * b


def divide(a, b):
    """
    Divide first number by second number.
    
    Args:
        a (int, float): Dividend (number to be divided)
        b (int, float): Divisor (number to divide by)
        
    Returns:
        float: Quotient of a divided by b
        
    Raises:
        ZeroDivisionError: If b is zero
        
    Example:
        >>> divide(15, 3)
        5.0
        >>> divide(10, 4)
        2.5
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


# Example usage and basic tests
if __name__ == "__main__":
    print("Basic Math Operations Demo")
    print("=" * 30)
    
    # Test addition
    print(f"Addition: 10 + 5 = {add(10, 5)}")
    
    # Test subtraction  
    print(f"Subtraction: 10 - 5 = {subtract(10, 5)}")
    
    # Test multiplication
    print(f"Multiplication: 10 * 5 = {multiply(10, 5)}")
    
    # Test division
    print(f"Division: 10 / 5 = {divide(10, 5)}")
    
    # Test with decimals
    print(f"Decimal Addition: 3.14 + 2.86 = {add(3.14, 2.86)}")
    
    # Test division by zero handling
    try:
        result = divide(10, 0)
    except ZeroDivisionError as e:
        print(f"Error handling: {e}")
