#!/bin/bash

mkdir -p data

# small_parallel_enja
git clone https://github.com/odashi/small_parallel_enja data/small_parallel_enja
rm -rf data/small_parallel_enja/.git

# wmt14_ende
mkdir -p data/wmt14_ende
URL=https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de
for d in train newstest2012 newstest2013 newstest2014 newstest2015; do
  for l in en de; do
    wget ${URL}/${d}.${l} -P data/wmt14_ende
  done
done
