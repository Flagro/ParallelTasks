Test run on 4 threads CPU:
```bash
Processing with 1 threads...
Threads: 1
Median Compression Time: 334.534 ms
Median GetData Time: 298.531 ms
Median GetSize Time: 0.001885 ms
Original Size: 4000000 bytes
Min Size: 875000 bytes
Max Size: 875000 bytes
Mean Size: 875000 bytes
Correctness: 1
----------------------------------------
Processing with 2 threads...
Threads: 2
Median Compression Time: 198.739 ms
Median GetData Time: 179.626 ms
Median GetSize Time: 0.0026615 ms
Original Size: 4000000 bytes
Min Size: 875000 bytes
Max Size: 875000 bytes
Mean Size: 875000 bytes
Correctness: 1
----------------------------------------
Processing with 3 threads...
Threads: 3
Median Compression Time: 135.407 ms
Median GetData Time: 123.062 ms
Median GetSize Time: 0.002777 ms
Original Size: 4000000 bytes
Min Size: 875000 bytes
Max Size: 875000 bytes
Mean Size: 875000 bytes
Correctness: 1
----------------------------------------
Processing with 4 threads...
Threads: 4
Median Compression Time: 120.715 ms
Median GetData Time: 108.605 ms
Median GetSize Time: 0.0025445 ms
Original Size: 4000000 bytes
Min Size: 875000 bytes
Max Size: 875000 bytes
Mean Size: 875000 bytes
Correctness: 1
----------------------------------------
```

To run the project:
```bash
mkdir build
cd build
cmake ..
make
./CompressedDataApp
```
