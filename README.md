# ParallelTask
This repository contains the implementation of tasks i got as homework for a job appliance for C++ developer with parallel CPU/GPU development tools stack.

## The tasks i was given
### Task1 - compression of integers with CPU
- Input data: sequence of 1 million integers from 1 to 100.
- The task is to implement compression and then decompression of this sequence.
- Implementation should be multithreaded and should be ran on CPU.

### Task2 - unique values with GPU
- The task is to generate (on CPU) a sequence of 10 millions int32 integers with at maximum 1000 unique values and implement an algorithm for obtaining all the values that are only occur once in the sequence on GPU.

## Comments on Task1:
- The formulation of the first task (based on the restrictions that all integers are in range [1...100]) looked like it meant to be a compression of a sorted sequence (or a compression where the order of a sequence could be lost) so i implemented both variants of the task: the folder **SetCompression** containts the solution for the case in which the order after decompression doesn't matter and folder **VectorCompression** contains the solution for the case in which the order matters.
- For integer compression in real world scenario i would use an existing solution with suiting license for commercial use and optimized for given architechture such as https://github.com/lemire/FastPFor and https://github.com/lemire/SIMDCompressionAndIntersection and would just dynamically link the library.
- The main idea behind the implementation is that there could be lots of ways to store the data (Huffman coding, VLE, Delta encoding, etc) but the interfaces needed for the task are the same. That's why the inheritance from DataFormat class was used. 
- The implementation tries to use as little space in RAM as possible, that's why move semantics are used and all the stl structures such as hash maps are avoided.
- The decompressed data is stored in RAM but the implementation allows it to be easily saved in binary files if needed.
- I also included the generation of JSON files so it would be easy to analyze/interpret/debug the results with a python scripts.
- The implementation uses stl containers instead of dynamic memory allocation with **new** but ensures that the capacity of the containers is restricted to take as little space in memory as possible.

## Comments on Task2:
- 

## Ways to improve:
- The implementations obviously could use some documentation.
- Currently the main.cpp files do have functions that could be moved to a new modules such as json generation module, aggregation functions, etc.
- Constants in main.cpp files could be passed as executable arguments.
