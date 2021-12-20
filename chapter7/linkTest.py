import sys 
import subprocess 
import os 


#***方法一
prog = r''' 
#include <iostream> 
using namespace std; 
int main(){ 
	cout<<"Hello World\n"; 
	return 1; 
	} 
''' 
'''
if not os.path.exists('Hello_world'): 
    f = open('Hello_world.cpp', 'w') 
    f.write(prog) 
    f.close() 
    #subprocess.call(["g++", "hello_world.cpp"]) 
    #tmp=subprocess.call("./a.out") 
    #print(tmp) 
    p = subprocess.Popen([r"/usr/bin/g++", "-Wall", "-o", "test", 'Hello_world.cpp'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
    p = subprocess.Popen(["./test"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.communicate()
'''
#***方法二 https://www.journaldev.com/31907/calling-c-functions-from-python
'''
from ctypes import *

so_file = 'foo.exe'
my_fuctions = CDLL(so_file)

print(type(my_fuctions))
print(my_fuctions)
'''
import os 
main = "foo.exe"
r_v = os.system(main) 
print (r_v )

main = "project1.exe"
f = os.popen(main)  
data = f.readlines()  
f.close()
print (data)