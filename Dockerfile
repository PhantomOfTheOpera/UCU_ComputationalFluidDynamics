FROM virialdi/test


RUN git clone --recursive https://github.com/arrayfire/arrayfire.git
RUN mkdir arrayfire/build
WORKDIR /arrayfire/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j16