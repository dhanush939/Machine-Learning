def input_matrices(order1,order2):
        a=[]
        for j in range(order1):
                row=[]
                for k in range(order2):
                        r=int(input("enter the number:"))
                        row.append(r)
                a.append(row)        
        return a
def multiplication(A,B):
    result = []
    for i in range(len(A)):          
        row = []
        for j in range(len(B[0])): 
            sum = 0
            for k in range(len(B)):  
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result
r1=int(input("enter no of rows :"))
c1=int(input("enter the no of columns "))
r2=int(input("enter no of rows :"))
c2=int(input("enter the no of columns "))
if c1 == r2 :
        k=input_matrices(r1,c1)
        print(f"matrix 1: {k}")
        k1=input_matrices(r2,c2)
        print(f"matrix 12: {k1}")
        r=multiplication(k,k1)
        print(f"multiplecation of above 2 matrices is {r}")

else:
      print("Multiplication is not possible due to inproper sizes ")