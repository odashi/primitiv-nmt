primitiv-nmt
============

Neural machine translation tool developed by [primitiv](https://github.com/primitiv/primitiv)


Prerequisites
-------------

- C++11 compiler
- primitiv
- protobuf3


Build/install
-------------

    $ git clone <this repository>
    $ cd <this repository>
    $ mkdir build
    $ cd build
    $ cmake .. [-DPRIMITIV_NMT_USE_CUDA=ON]
    $ make -j<threads>
    $ [sudo make install]

Usage
-----

See [sample.sh](sample.sh).
