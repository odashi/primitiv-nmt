CFLAGS=-std=c++11
LIBS=-lprimitiv
HDRS=utils.h vocab.h
encdec_attention_SRCS=encdec_attention.cc

all:

encdec_attention: ${encdec_attention_SRCS} ${HDRS}
	g++ ${CFLAGS} -o $@ ${encdec_attention_SRCS} ${LIBS}
