##linux的安装包
#build code 
#!/bin/bash

mkdir build
cd build
rm -rf *
cmake .. -DBUILD_PLATFORM=x86_64
make
sudo make install


