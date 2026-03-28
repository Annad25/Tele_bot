# Python FAQ & Advanced Concepts Guide

## Page 1: Python Decorators - Concepts and Examples

### 1.1 What is a Decorator?
A decorator in Python is a design pattern that allows a user to add new functionality to an existing object (a function, method, or class) without modifying its structure. Decorators are highly used in Python for cross-cutting concerns like logging, access control, instrumentation, or caching.

### 1.2 The Syntax
Decorators are usually called before the definition of a function using the `@` symbol. Under the hood, they are higher-order functions that take a function as an argument and return a new function.

```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()
```

### 1.3 Decorators with Arguments
To build a decorator that accepts arguments, you need a three-level nested function. The outermost function handles the arguments, the middle handles the decorated function, and the innermost acts as the execution wrapper.

## Page 2: Understanding the Global Interpreter Lock (GIL)

### 2.1 What is the GIL?
The Python Global Interpreter Lock (GIL) is a mutex (or a lock) that allows only one thread to hold the control of the Python interpreter. This means that only one thread can be in a state of execution at any point in time in a given Python process.

### 2.2 Why Does the GIL Exist?
The GIL was adopted in CPython (the reference implementation of Python) because Python's memory management is not thread-safe. Python uses reference counting for garbage collection, and without the GIL, concurrent threads could cause race conditions, leading to memory leaks or incorrect deletion of objects still in use.

### 2.3 Impacts on Concurrency
The GIL is a bottleneck for CPU-bound multi-threaded programs. Even if a computer has multiple cores, a Python program using standard multi-threading cannot fully utilize them for CPU-bound tasks.
- **I/O-bound tasks:** Multi-threading is highly effective because threads release the GIL when waiting for I/O (network, file system).
- **CPU-bound tasks:** Multi-processing (using the `multiprocessing` module) is recommended because each process gets its own memory space and its own GIL.

## Page 3: Data Structures: Lists vs. Tuples

### 3.1 Python Lists
Lists are ordered, mutable collections of items. You can add, remove, or change items after the list is created.
- Syntax: `my_list = [1, 2, 3]`
- Memory footprint: Larger, as lists allocate extra memory to allow for fast `.append()` operations.
- Best use-cases: When you have a collection of items that might change, such as items in a shopping cart or a stack of active processes.

### 3.2 Python Tuples
Tuples are ordered, immutable collections. Once created, their size and elements cannot be modified (though if a tuple contains mutable objects like a list, those inner objects can be changed).
- Syntax: `my_tuple = (1, 2, 3)`
- Memory footprint: Smaller. They are statically sized.
- Best use-cases: For heterogeneous data structures (like a row in a database), for ensuring data integrity, or as keys in dictionaries (since keys must be hashable and immutable).

### 3.3 Key Differences Summary
| Feature | List | Tuple |
| --- | --- | --- |
| Mutability | Mutable | Immutable |
| Syntax | Square brackets `[]` | Parentheses `()` |
| Memory | Consumes more memory | Consumes less memory |
| Speed | Slower iteration | Faster iteration |

## Page 4: Environment Management: Virtual Environments

### 4.1 Why Virtual Environments?
A virtual environment is a self-contained directory tree that contains a Python installation for a particular version of Python, plus a number of additional packages. It solves the "dependency hell" problem by ensuring that different projects can use different versions of the same library without conflicts.

### 4.2 Creating a Virtual Environment
The standard way to create virtual environments in modern Python is using the built-in `venv` module.
`python3 -m venv myenv`

### 4.3 Activation and Usage
To use the virtual environment, you must activate it.
- **Windows:** `myenv\Scripts\activate`
- **macOS/Linux:** `source myenv/bin/activate`
Once activated, any `pip install` commands will place packages inside the isolated environment rather than the systemic Python paths.

### 4.4 Managing Dependencies with Requirements list
It is best practice to export your dependencies into a `requirements.txt` file using `pip freeze > requirements.txt`. Other developers can install identical dependencies by running `pip install -r requirements.txt`.
