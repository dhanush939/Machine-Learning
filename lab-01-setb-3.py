def compare_lists(list1,list2):
    # taking an linear search .
    common=[]
    for i in list1:
        if i in list2 :
            common.append(i)
    return common
list1=list(map(int,input("enter the numbers").split()))        
list2=list(map(int,input("enter the numbers ").split()))
k=compare_lists(list1,list2)
if len(k)==0 :
    print("none")
else:
    print(k)