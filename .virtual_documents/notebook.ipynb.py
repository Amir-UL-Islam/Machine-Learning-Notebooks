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
a, b, c = tup
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
a_list.pop(3)
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






i = 0

for i in 





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









