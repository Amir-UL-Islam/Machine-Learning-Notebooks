# Defining the Employee Class
class employee:
    # defining the properties and assigning with NONE
    # properties can not be undifined
    Id = None
    Salary = None
    Department = None
    
    # There are two ways to assign values to properties of a class.
    # 1. Assign values when defining the class.
    # 2. Assign values in the main code.


# Creating a object name steve
steve = employee()
  
# Assigning the value to the steve's properties. steve an object of employee class.
steve.Id = 1020
steve.Salary = 2590
steve.Department = 'Engineer'

# Creating a new propertey and assign its value OUT side the class scope
steve.Title = 'Manager'

# printing the value of Properties
print('Steve ID is {}'.format(steve.Id))
print('And salary is {}'.format(steve.Salary))
print('Department {}'.format(steve.Department))
print('He is also a {}'.format(steve.Title))



class Employee:
    
    def __init__(self):
        self.name = 'defalt peramiter' # Here self.name is just a data member.

    def print_function(self):
        print(self.name)
    
objt = Employee()

objt.print_function()


class Addition:
    first = 0
    second = 0
    third = 0
    
    def __init__(self, f, s, t):
        self.first = f
        self.second = s
        self.third = t
    
    def display(self):
        print('First Number: ' + str(self.first))
        print('Second Number: ' + str(self.second))
        print('Third Number: ' + str(self.third))
        print('Answer: ' + str(self.answer))
    
    def calculate(self):
        self.answer = self.first+self.second+self.third
        
objt = Addition(5, 3, 2)

objt.calculate()

objt.display()



# Defining the Employee Class
class employee:
    
    # defining the properties and assigning with NONE
    # properties can not be undifined
    #Initializing the properties
    def __init__(self, Id = None, Salary = None, Department = None):
        self.Id = Id
        self.Salary = Salary
        self.Department = Department
        
        # There are two ways to assign values to properties of a class.
        # 1. Assign values when defining the class.
        # 2. Assign values in the main code.
        
# Creating a object name steve with default parameters
steve = employee(112, 334433, 'Executive')
 
# printing the value of Properties
print('Steve ID is {}'.format(steve.Id))
print('And salary is {}'.format(steve.Salary))
print('Department {}'.format(steve.Department))

# The initializer is automatically called when an object of the class is created. 
# It is important to define the initializer with complete parameters to avoid any errors.
# Similar to methods, initializers also have the provision for optional parameters.



class Player:
    teamName = 'Liverpool'  # class variables

    def __init__(self, name):
        self.name = name  # creating instance variables

    @classmethod
    def getTeamName(cls):
        return cls.teamName


print(Player.getTeamName())



class BodyInfo:

    @staticmethod
    def bmi(weight, height):
        return weight / (height**2)


weight = 75
height = 1.8
print(BodyInfo.bmi(weight, height))



class Employee:
    def __init__(self, ID, salary):
        # all properties are public
        self.ID = ID
        self.salary = salary

    def displayID(self):
        print("ID:", self.ID)


Steve = Employee(3789, 2500)
Steve.displayID()
print(Steve.salary)



class Employee:
    def __init__(self, ID, salary):
        self.ID = ID
        self.__salary = salary  # salary is a private property


Steve = Employee(3789, 2500)
print("ID:", Steve.ID)
print("Salary:", Steve.__salary)  # this will cause an error



class Employee:
    def __init__(self, ID, salary):
        self.ID = ID
        self.__salary = salary  # salary is a private property

    def displaySalary(self):  # displaySalary is a public method
        print("Salary:", self.__salary)

    def __displayID(self):  # displayID is a private method
        print("ID:", self.ID)


Steve = Employee(3789, 2500)
Steve.displaySalary()
Steve.__displayID()  # this will generate an error


class Employee:
    def __init__(self, ID, salary):
        self.ID = ID
        self.__salary = salary  # salary is a private property


Steve = Employee(3789, 2500)
print(Steve._Employee__salary)  # accessing a private property



class User:
    def __init__(self, username=None):  # defining initializer
        self.__username = username

    def setUsername(self, x):
        self.__username = x

    def getUsername(self):
        return (self.__username)


Steve = User('opu')
print('Before setting:', Steve.getUsername())
Steve.setUsername('Amir')
print('After setting:', Steve.getUsername())


class User:
    def __init__(self, userName=None, password=None):
        self.userName = userName
        self.password = password

    def login(self, userName, password):
        if ((self.userName.lower() == userName.lower())
                and (self.password == password)):
            print("Access Grantedget_ipython().getoutput("")")
        else:
            print("Invalid Credentialsget_ipython().getoutput("")")


Steve = User("Steve", "12345")
Steve.login("steve", "12345")
Steve.login("steve", "6789")
Steve.password = "6789"
Steve.login("steve", "6789")


class User:
    def __init__(self, userName=None, password=None):
        self.__userName = userName
        self.__password = password

    def login(self, userName, password):
        if ((self.__userName.lower() == userName.lower())
                and (self.__password == password)):
            print(
                "Access Granted against username:",
                self.__userName.lower(),
                "and password:",
                self.__password)
        else:
            print("Invalid Credentialsget_ipython().getoutput("")")


# created a new User object and stored the password and username
Steve = User("Steve", "12345")
Steve.login("steve", "12345")  # Grants access because credentials are valid

# does not grant access since the credentails are invalid
Steve.__password = '6789'
Steve.login("steve", "6789")


                


class Vehicle:
    def __init__(self, make=None, color=None, model=None):
        self.make = make
        self.color = color
        self.model = model

    def printDetails(self):
        print("Manufacturer:", self.make)
        print("Color:", self.color)
        print("Model:", self.model)


class Car(Vehicle):
    def __init__(self, make, color, model, doors):
        # calling the constructor from parent class
        Vehicle.__init__(self, make, color, model)
        self.doors = doors

    def printCarDetails(self):
        self.printDetails()
        print("Doors:", self.doors)

class Jeep(Car):
    def __init__(self, make, color, model, doors, windows):
        # calling the constructor from parent class
        Car.__init__(self, make, color, model, doors)
        self.windows = windows
    
    def printJeepDetails(self):
        self.printCarDetails()
        print('Windows:', self.windows)


obj1 = Car("Suzuki", "Grey", "2015", 4)
obj2 = Jeep('Nissan', 'Space Grey', 'gt-r r35', 2, 4)
obj1.printCarDetails()
obj2.printJeepDetails()


class Vehicle:  # defining the parent class
    fuelCap = 90


class Car(Vehicle):  # defining the child class
    fuelCap = 50

    def display(self):
        # accessing fuelCap from the Vehicle class using super()
        print("Fuel cap from the Vehicle Class:", super().fuelCap)

        # accessing fuelCap from the Car class using self
        print("Fuel cap from the Car Class:", self.fuelCap)


obj1 = Car()  # creating a car object
obj1.display()  # calling the Car class method display()



class Vehicle:  # defining the parent class
    def display(self):  # defining display method in the parent class
        print("I am from the Vehicle Class")


class Car(Vehicle):  # defining the child class
    # defining display method in the child class
    def display(self):
        super().display()
        print("I am from the Car Class")


obj1 = Car()  # creating a car object
obj1.display()  # calling the Car class method display()


class Vehicle:
    def __init__(self, make=None, color=None, model=None):
        self.make = make
        self.color = color
        self.model = model

    def printDetails(self):
        print("Manufacturer:", self.make)
        print("Color:", self.color)
        print("Model:", self.model)


class Car(Vehicle):
    def __init__(self, make, color, model, doors):
        super().__init__(make, color, model)
        self.doors = doors

    def printCarDetails(self):
        self.printDetails()
        print("Door:", self.doors)


obj1 = Car("Suzuki", "Grey", "2015", 4)
obj1.printCarDetails()


class Account:  # parent class
    def __init__(self, title=None, balance=0):
        self.title = title
        self.balance = balance
    
    # withdrawal method subtracts the amount from the balance
    def withdrawal(self, amount):
        self.balance = self.balance - amount # amount is not a Account class propertiy. It's a property of the methods define within Account class.
    
    # deposit method adds the amount to the balance
    def deposit(self, amount):
        self.balance = self.balance + amount
    
    # this method just returns the value of balance
    def getBalance(self):
        return self.balance


class SavingsAccount(Account):
    def __init__(self, title=None, balance=0, interestRate=0):
        super().__init__(title, balance)# I have already initialized title and balance before. So, I don't need to initialize again.
        self.interestRate = interestRate
    
    # computes interest amount using the interest rate
    def interestAmount(self):
        return (self.balance * self.interestRate / 100)


obj1 = SavingsAccount("Steve", 5000, 10)
print("Initial Balance:", obj1.getBalance())
obj1.withdrawal(1000)
print("Balance after withdrawal:", obj1.getBalance())
obj1.deposit(500)
print("Balance after deposit:", obj1.getBalance())
print("Interest on current balance:", obj1.interestAmount())



class Rectangle():

    # initializer
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.sides = 4

    # method to calculate Area
    def getArea(self):
        return (self.width * self.height)


class Circle():
    # initializer
    def __init__(self, radius=0):
        self.radius = radius
        self.sides = 0

    # method to calculate Area
    def getArea(self):
        return (self.radius **2 * 3.142)


shapes = [Rectangle(6, 10), Circle(7)]
print("Sides of a rectangle are", str(shapes[0].sides))
print("Area of rectangle is:", str(shapes[0].getArea()))

print("Sides of a circle are", str(shapes[1].sides))
print("Area of circle is:", str(shapes[1].getArea()))



class Shape:
    def __init__(self):  # initializing sides of all shapes to 0
        self.sides = 0

    def getArea(self):
        pass


class Rectangle(Shape):  # derived from Shape class
    # initializer
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.sides = 4

    # method to calculate Area
    def getArea(self):
        return (self.width * self.height)


class Circle(Shape):  # derived from Shape class
    # initializer
    def __init__(self, radius=0):
        self.radius = radius

    # method to calculate Area
    def getArea(self):
        return (self.radius * self.radius * 3.142)


shapes = [Rectangle(6, 10), Circle(7)]
print("Area of rectangle is:", str(shapes[0].getArea()))
print("Area of circle is:", str(shapes[1].getArea()))



class Shape:
    def __init__(self):  # initializing sides of all shapes to 0
        self.sides = 0

    def getArea(self):
        return self.sides
    
obj = Shape()
print(obj.getArea())

class Rectangle(Shape):  # derived form Shape class
    # initializer
    def __init__(self, width=0, height=0):
        self.width = width
        self.height = height
        self.sides = 4

    # method to calculate Area
    def getArea(self):
        return (self.width * self.height)


class Circle(Shape):  # derived form Shape class
    # initializer
    def __init__(self, radius=0):
        self.radius = radius

    # method to calculate Area
    def getArea(self):
        return (self.radius * self.radius * 3.142)


shapes = [Rectangle(6, 10), Circle(7)]
print("Area of rectangle is:", str(shapes[0].getArea()))
print("Area of circle is:", str(shapes[1].getArea()))


# Parent Class
class Shape:
    sname = "Shape"

    def getName(self):
        return self.sname


# child class
class XShape(Shape):
    # initializer
    def __init__(self, name):
        self.xsname = name

    def getName(self):  # overriden method
        return (super().getName() + ", " + self.xsname)


circle = XShape("Circle")
print(circle.getName())



a = 5
b = 6

c = a + b
print(c)


class Student:

    def __init__(self,m1,m2):
        self.m1 = m1
        self.m2 = m2

    def __add__(self, other):
        m1 = self.m1 + other.m1
        m2 = self.m2 + other.m2
        s3 = Student(m1,m2)

        return  s3

    def __gt__(self, other):
        r1 = self.m1 + self.m2
        r2 = other.m1 + other.m2
        if r1 > r2:
            return True
        else:
            return False

    def __str__(self):
        return '{} {}'.format( self.m1,self.m2)


s1 = Student(58, 69)
s2 = Student(69, 65)



s3 = s1 + s2

if s1 > s2:
    print("s1 wins")
else:
    print("s2 wins")

a = 9
print(a.__str__())

print(s2)


class Complex:
    def __init__(self, real=None, imag=None):
        self.real = real
        self.imag = imag

    def __add__(self, other):  # overloading the `+` operator
        temp = Complex(self.real + other.real, self.imag + other.imag)
        return temp

    def __sub__(self, other):  # overloading the `-` operator
        temp = Complex(self.real - other.real, self.imag - other.imag)
        return temp


obj1 = Complex(3, 7)
obj2 = Complex(2, 5)

obj3 = obj1 + obj2
obj4 = obj1 - obj2

print('real of obj1:', obj1.real)
print('real of obj1:', obj1.imag)
print('real of obj1:', obj2.real)
print('real of obj1:', obj2.imag)

print('\n')

print("real of obj3:", obj3.real)
print("imag of obj3:", obj3.imag)
print("real of obj4:", obj4.real)
print("imag of obj4:", obj4.imag)



class Dog:
    def Speak(self):
        print("Woof woof")

class Cat:
    def Speak(self):
        print("Meow meow")

class AnimalSound:
    def Sound(self, animal):
        animal.Speak()

sound = AnimalSound()
dog = Dog()
cat = Cat()
# dog.Speak()
# cat.Speak()
sound.Sound(dog)
sound.Sound(cat)





class Animal:
    def __init__(self, name=None, sound=None):
        self.name = name
        self.sound = sound
    
    def Animal_details(self):
        print(self.name)
        print(self.sound)

class Dog(Animal):
    def __init__(self, name=None, sound=None, family=None):
        super().__init__(name, sound)
        self.family = family

    def Animal_details(self):
        super().Animal_details()
        print(self.family)

class Sheep(Animal):
    def __init__(self, name=None, sound=None, color=None):
        super().__init__(name, sound)
        self.color = color

    def Animal_details(self):
        super().Animal_details()
        print(self.color)


d = Dog("Pongo", "Woof Woof", "Husky")
d.Animal_details()
print(" ")
s = Sheep("Billy", "Baaa Baaa", "White")
s.Animal_details()
