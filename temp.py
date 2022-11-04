def getsum(a):
 
    r = 0
    sum = 0
    while (a > 0) :
        r = a % 10
        sum = sum + r
        a = a // 10
     
    return sum
 
# This function creates
# the array of valid numbers
def value(a, b, c):
 
    x = 0
    q = 0
    w = 0
    v = []
 
    for i in range(1, 82) :
 
        # this computes s(x)^a
        no = pow(i, a)
 
        # this gives the result
        # of equation
        no = b * no + c
 
        if (no > 0 and no < 1000000000) :
            x = getsum(no)
             
            # checking if the sum same as i
            if (x == i) :
 
                # counter to keep track
                # of numbers
                q += 1
 
                # resultant array
                v.append(no)
                w += 1
                 
    # prints the number
    for i in range(len(v)) :
        print(v[i], end = " ")
 
# Driver Code
if __name__ == "__main__":
     
    a = 1
    b = 1
    c = 1
 
    # calculate which value
    # of x are possible
    value(a, b, c)
 
# This code is contributed
# by ChitraNayal