#!/bin/bash

BIN=./build/primitiv_nmt
CORPUS=./sample/corpus
MODEL=./sample/model

mkdir -p sample

if [ ! -e corpus/train.en ]; then
  git clone https://github.com/odashi/small_parallel_enja ${CORPUS}
  rm -rf ${CORPUS}/.git
fi

mkdir -p sample/model

${BIN}/make_vocab 4100 ${CORPUS}/train.en ${MODEL}/vocab.en
${BIN}/make_vocab 4900 ${CORPUS}/train.ja ${MODEL}/vocab.ja

for f in train dev; do
  ${BIN}/make_corpus 1 20 \
    ${CORPUS}/${f}.{en,ja} \
    ${MODEL}/vocab.{en,ja} \
    ${MODEL}/corpus.${f}
done

EMBED=512
HIDDEN=512
BATCH=64
LEARNING_RATE=0.0001 # Adam
EPOCHS=10
GPUID=0

${BIN}/train \
  ${MODEL}/corpus.{train,dev} \
  ${MODEL}/vocab.{en,ja} \
  ${MODEL}/model \
  ${EMBED} ${HIDDEN} ${BATCH} ${LEARNING_RATE} ${EPOCHS} ${GPUID}

${BIN}/translate \
  ${MODEL}/vocab.{en,ja} \
  ${MODEL}/model \
  $(cat ${MODEL}/model/best.epoch) \
  ${GPUID} \
  < ${CORPUS}/test.en \
  | tee ${MODEL}/model/hyp
