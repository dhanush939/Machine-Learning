def count_vowels(s):
     return sum(1 for ch in s if ch.lower() in("aeiou") )
def count_consonents(s):
     return sum(1 for ch in s if ch.lower() not in("aeiou"))
s=input("enter the string :")
c1=count_consonents(s)
c2=count_vowels(s)
print(f"the count of the consonents are : {c1} \n the count of the vowels are :{c2} ")
