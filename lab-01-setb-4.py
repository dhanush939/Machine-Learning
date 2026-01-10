def input_matrix(num1,num2):
    a=[]
    for i in range(num1):
        row=[]
        for j in range(num2):
            r=int(input(f"enter the number of {i} and {j}"))
            row.append(r)
        a.append(row)  
    return a
def transpose(matrix):
    rows=len(matrix)
    cols=len(matrix[0])
    t=[]
    for i in range(cols):
        row=[]
        for j in range(rows):
            row.append(matrix[j][i])
        t.append(row)
    return t    
row=int(input("enter the row "))
col=int(input("enter the columns :"))
k=input_matrix(row,col)
print(f"the original matrix is :{k} ")
p=transpose(k)
print(f"the transpose of the matrix is :  {p}")