def functionThatThrowsSyntaxError():
    some_list = [1,2,3,4,5]
    list_object = [(x**2 for x in some_list] 
    print("There's no possible way for this line to contain an error")

def functionThatComputesSomething(a,b):
    print("Computing the value of " +str(a)+" + " + str(b))
    print("The outputted answer is",a+b)

def main():
    functionThatComputesSomething(3,4)

if __name__ == '__main__':
    main()

