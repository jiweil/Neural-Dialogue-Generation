import sys
A=open(sys.argv[1],"r")
lines=A.readlines();
print("num of lines "+str(len(lines)))
Dict={}
count=0
for line in lines:
    t1=line.find(" ")
    t2=line.find(" ",t1+1)
    t3=line.find(" ",t2+1)
    line=line[t3+1:].strip()
    if Dict.has_key(line)is False:
        Dict[line]=0;
    Dict[line]=Dict[line]+1;
    count=count+1
    if count==1000000:
        break

A=open(sys.argv[2],"w")

sort=sorted(Dict.items(), key=lambda x: x[1],reverse=True)
t=0
for item in sort:
    """
    A.write(str(item[1])+" "+str(item[0])+"\n")
    """
    A.write(str(item[1])+"|"+str(item[0])+"\n")
    t=t+1;
    if item[1]<=50:break
