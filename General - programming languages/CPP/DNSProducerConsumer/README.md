# DNS resolver ThreadPool Producer-Consumer based on CPP

![alt tag](https://github.com/orel1212/Portfolio/blob/main/General%20-%20programming%20languages/CPP/DNSProducerConsumer/%E2%80%8F%E2%80%8Farchitecture.PNG)

## How To Run
run ./sources/main.cpp via c++ compiler/IDE <br>
Instructions how to run the program:  <br>
1.export LD_LIBRARY_PATH=path where the directory is  <br>
2.move the input files from "input" directory to the main directory(where all the cpp files are)  <br>
3.run the bash command:  <br>
g++ -o main main.cpp -D_POSIX_C_SOURCE=200809 -std=c++11 -Wall -pedantic -O2 -g -march=native -L. -lpthread -lmyutil  <br>

4.run the bash command  <br>
./main n1.t names1.txt n2.t names2.txt names3.txt names4.txt n3.t names5.txt n4.t results.txt  <br>

to see that n1.t,n2.t,n3.t,n4.t not exist and the program just jump over it to the next files.  <br>

-------------------------------------------------------------  <br>

Notes:  <br>

1.the program is bounded to 10 files(without the ./run and output file, which in total is 12)  <br>
2.the program will create Requesters according to the num of files (without ./run and output) which are valid,if there will be no valid files the program will exit.  <br>
3. num of resolvers currently is 5. <br>
4. there is an assumption that output file is empty(if exists).  <br>
