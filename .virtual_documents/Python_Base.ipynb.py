tup = 3, 4, 6
tup


tup_compli = (3, 4, 8), (3, 8)
print(tup_compli)

# Concatenate and multiply tuples
concat_tup = (4, None, 'foo') + (6, 0) + ('bar',)
print(concat_tup)

# Converting a sequence or iterator to a tuple
lis_t = [2, 3, 'lala']
new_tup = tuple(lis_t)
print(new_tup)

# Accessing with brackets[]
new_tup[2]


# Immutable Tuple
tup = tuple(['foo', [1, 2], True])
# tup[2] = False(First Uncomment This pice of code)

# Mutable objects of Tuple
tup[1].append(3)
tup


tup = (4, 5, 6)
a, b, c = tup # assigning a = 4, b = 5, c = 6
print(a)

# Nasted tuple
tup = 4, 5, (6, 7)
a, b, (c, d) = tup
(c, d)




values = 1, 2, 3, 4, 5
another_value = 3, 5, 7, 3, 2
a, b, *rest = values
a, b, *_ = another_value


print(_)
print(rest)


seq = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
seq
for a, b, c in seq:
    print('a = {0}, b = {1}, c = {2}'.format(a, b, c))
   


a_list = [2, 3, 7, None]
a_list
print(a_list[2])



a_list = ['a', 'b', 'c', 'd', 'e']
a_list.append('f')
print(a_list)
a_list.insert(2, 'g')
print(a_list)
a_list.pop()
print(a_list)
a_list.remove('a')
print(a_list)


print('a' in a_list)

print('a' not in a_list)


foo_list = [4, None, 'foo'] + [7, 8, (2, 3)]
print(foo_list)

# The Extend method
x = [4, None, 'foo']
x.extend([7, 8, (2, 3), [88, 45, 90, 234]])
print(x)

# More faster way for large file 
everything = [4, None, 'foo']
list_of = [[7, 8, (2, 3)], ['a', 'b', 'c', 'd', 'e']]
for chunk in list_of:
    everything.extend(chunk)


b = ['saw', 'small', 'He', 'foxes', 'six']
b.sort()
print(b)
b.sort(key=len)
print(b)


any_seq = [7, 2, 4, 6, 9, 11, 5, 8]
print(any_seq[2:5])

#In case of Start or Stop is not present
print(any_seq[:5])

print(any_seq[2:])


print(any_seq[-1:])
print(any_seq[-6: -2])


empty_dicto = {}

# Define a Dict
dict = {'key1':'Value 1',
        'a':'any',
        '7':['apple', 'mango', 'oil'],
        'dummy':'value'}

# Access and inseting a new key, value
print(dict['a'])

# Inserting
dict['a'] = 'can be anything'
print(dict)

# New key and Value
dict['b'] = 'not yet'
print(dict)

# Checking the values with same syntax as list or tuple
'b' in dict

# Deleting the Key and Values using del and pop 
del dict['key1']
print(dict)

rest = dict.pop('7')
print(rest)
print(dict)



print(dict.keys())
print(dict.values())


print(dict)

# Define an other dict
another_dict = {'ola':['me', '7'],
                'dash':'foo'}
# merging to dict
dict.update(another_dict)


# Creating dicts from sequence
seq1 = ['one', 'two', 'three']
seq2 = ['non', 'nothing', 'never']
mapping = {}
for key, value in zip(seq1, seq2):
    mapping[key] = value
print(mapping)


# Using string type as keys
print(hash('string'))

# Usint tuple type as keys
print(hash((1, 2, (2, 3))))



# Try uncomment under this line of code
# hash((1, 2, [2, 3])) 


seta = set([2, 2, 4, 1, 3, 3])
print(seta)


# Union operation
a = {2, 3, 4}
b = {5, 6, 7, 2}
c = a.union(b)
print(c)

# Intersection operation
d = a.intersection(b)
print(d)


# Defining a string
string = ['a', 'for', 'apple', 'b', 'foor', 'bat']

# Defining the comprehension
result = [x.upper() for x in string]
print(result)



# Defining a string
string = ['a', 'for', 'apple', 'b', 'foor', 'bat']

# Defining the comprehension
result = [x.upper() for x in string if len(x) > 3]
print(result)


# Creating a nseted list
data = [['a', 2.35], ['b', -3.4], ['c', .35], ['d', 33], ['e', 34], ['f', -3.4], ['g', 33], ['h', 33]]
after_opperation = []

# creatig a list with every first elements
every_first_elements = [a for a, b in data]
print(every_first_elements)

# creating a list with every second elements
every_second_elements = [b for a, b in data if b>22]
print(every_second_elements)


# Creating a function
def function_name(perameter):
    print(perameter)

# Passing the arguments while calling a the function
function_name('arguments')



# Creating a function with positional Arguments
def function_name(a, b, c):
    print(a, b, c)

# Calling the function
function_name(1, 4, 6)

# Using keyward Arguments
def function_name(a, b, c):
    print(a, b, c)
    
# caling the function Wiht key wards
function_name(b=2, a=3, c=4) # order dosen't matter

# For default values
def function_name(a, b, c='default'):
    print(a, b, c)

function_name(1, 2)# passing only Two values


# Errors
def function_e(a, b, c):
    print(a, b, c)

function_e(1, b=3, 2)



# Errors
def function_er(a, b, c):
    print(a, b, c)
    
function_er(1, b=3, a=4)



# Creating a list
list_t = [0]
print(list_t)

# With Repeated Elements
list_at = [1]*10
print(list_at)

# It also work with strings and tuple
tupl = (0, 1)*10
print(tupl)

# With strings
st = 'what \n'*2 # \n for new line.
print(st)


import re

for _ in range(int(input())):
    u = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', u)
        assert re.search(r'\d\d\d', u)
        assert not re.search(r'[^a-zA-Z0-9]', u)
        assert not re.search(r'(.)\1', u)
        assert len(u) == 10
    except:
        print('Invalid')
    else:
        print('Valid')


# Open "alice.txt" and assign the file to "file"
with open('datasets/alice.txt') as file:
    text = file.read()

n = 0
for word in text.split():
    if word.lower() in ['cat', 'cats']:
        n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))


import contextlib
@contextlib.contextmanager
def me():
    print('Say, Higet_ipython().getoutput("')")
    yield 42
    print('No You Suck')


with me() as me:
    print('I hate to say {}'.format(me))


@contextlib.contextmanager
def open_read_only(filename):
    """Open a file in read-only mode.

  Args:
    filename (str): The location of the file to read

  Yields:
    file object
  """
    read_only_file = open(filename, mode='r')
    
  # Yield read_only_file so it can be assigned to my_file
    yield read_only_file
    
  # Close read_only_file
    read_only_file.close()

with open_read_only('datasets/alice.txt') as my_file:
    print(my_file.read())


# Use the "stock('NVDA')" context manager gives 10 stock price by calling .price() method
# and assign the result to the variable "nvda"
with stock('NVDA') as nvda:
    
  # Open "NVDA.txt" for writing as f_out
    with open('NVDA.text', 'w') as f_out:
        for _ in range(10):
            value = nvda.price()
            print('Logging ${:.2f} for NVDA'.format(value))
            f_out.write('{:.2f}\n'.format(value))


def in_dir(directory):
    """Change current working directory to `directory`,
    allow the user to run some code, and change back.

    Args:
    directory (str): The path to a directory to work in.
    """
    current_dir = os.getcwd()
    os.chdir(directory)

  # Add code that lets you handle errors
    try:
        yield
  # Ensure the directory is reset,
  # whether there was an error or not
    finally:
        os.chdir(current_dir)


# Adding function references to the function map
import numpy as np
import pandas as pd
import random

function_map = {
  'mean': np.mean,
  'std': np.std,
  'minimum': np.min,
  'maximum': np.max,
    'foo': 'not me'
}

def load_data():
    columns = ['height', 'weight']
    data = [[72.1 ,  198 ], [69.8, 204], [69.8 , 164]]
    return pd.DataFrame(data, columns=columns)
    

# data = load_data()
print(data)


# def get_user_input():
#     x = input()
#     return x

# func_name = get_user_input()


# # # Call the chosen function and pass "data" as an argument
# function_map[func_name](data)


a = 5

def foo(value):
    def bar():
        print(value) # will print non-local variable a.
    return bar

function = foo(a)

function()


type(function.__closure__)


len(function.__closure__)


function.__closure__[0].cell_contents


del(a)


function()


len(function.__closure__)


function.__closure__[0].cell_contents


def double_args(func):
    def wrapper(a, b):
        return func(a*2, b*2)
    return wrapper

def multiply(a, b):
    return a * b

multiply = double_args(multiply)
multiply(1, 5)


def double_args(func):
    def wrapper(a, b):
        return func(a*2, b*2)
    return wrapper

@double_args
def multiply(a, b):
    return a * b

multiply(1, 5)


import time

def timer(func):
    """A decorator to print how much time a function take to run.

    Args:
    func(callable)

    Returns:
    callable
    """
    def wrapper(*args, **kargs):
        s_time = time.time()
        
        result = func(*args, **kargs)
        
        total_time = time.time() - s_time
        print('The Function {} took {} to run.'.format(func.__name__, total_time))
    return wrapper


@timer
def sleep_n_seconds(n):
    time.sleep(n)
sleep_n_seconds(2)


def memorize(func):
    catch = {}
    def wrapper(*args, **kwargs):
        if (args, kwargs) not in catch:
            catch[(args, kwargs)] = func(*args, **kwargs)
        return catch[(args, kwargs)]
    return wrapper



@memorize
def show_function(a, b):
    print('sleeping.....')
    time.sleep(3)
    return a + b

show_function(3, 2)


def print_return_type(func):
    # Define wrapper(), the decorated function
    def wrapper(*args, **kwargs):
        # Call the function being decorated
        result = func(*args, **kwargs)
        print('{}() returned type {}'.format(func.__name__, type(result)))
        
        return result
    # Return the decorated function
    return wrapper
  
@print_return_type
def foo(value):
    return value
  
print(foo(42))
print(foo([1, 2, 3]))
print(foo({'a': 42}))


def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        
        # Call the function being decorated and return the result
        return func
    wrapper.count = 0
    # Return the new decorated function
    return wrapper

# Decorate foo() with the counter() decorator
@counter
def foo():
    print('calling foo()')
  
foo()
foo()
foo()

print('foo() was called {} times.'.format(foo.count))
