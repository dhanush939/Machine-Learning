import random
list_100=[random.randint(100,150) for _ in range(100)]
mean_v=sum(list_100)/len(list_100)
print(f"mean of the generated data is : {mean_v}")
list_100.sort()
median=(list_100[49]+list_100[50])/2
print(f"median is : {median}")
mode=max(set(list_100),key=list_100.count)
print(mode)