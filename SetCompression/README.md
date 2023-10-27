Test run on 4 threads CPU:
```bash
Processing with 1 threads...
Threads: 1
Median Compression Time: 5.14324 ms
Median GetData Time: 8.52012 ms
Median GetSize Time: 0.001584 ms
Min Size: 412 bytes
Max Size: 412 bytes
Mean Size: 412 bytes
Correctness: 1
----------------------------------------
Processing with 2 threads...
Threads: 2
Median Compression Time: 3.42833 ms
Median GetData Time: 6.95144 ms
Median GetSize Time: 0.0013555 ms
Min Size: 412 bytes
Max Size: 412 bytes
Mean Size: 412 bytes
Correctness: 1
----------------------------------------
Processing with 3 threads...
Threads: 3
Median Compression Time: 2.62285 ms
Median GetData Time: 5.88236 ms
Median GetSize Time: 0.0013575 ms
Min Size: 412 bytes
Max Size: 412 bytes
Mean Size: 412 bytes
Correctness: 1
----------------------------------------
Processing with 4 threads...
Threads: 4
Median Compression Time: 1.79572 ms
Median GetData Time: 4.93888 ms
Median GetSize Time: 0.0013325 ms
Min Size: 412 bytes
Max Size: 412 bytes
Mean Size: 412 bytes
Correctness: 1
```

To run the project:
```bash
mkdir build
cd build
cmake ..
make
./CompressedDataApp
```
