CFLAGS=-std=c++11

base_HDRS=utils.h vocab.h
base_PROTOS=messages.proto
base_PB_CCS=messages.pb.cc
base_LIBS=-lprimitiv -lprotobuf

encdec_attention_SRCS=encdec_attention.cc ${base_PB_CCS}

all:

base_pbs: ${base_PROTOS}
	protoc --cpp_out=. ${base_PROTOS}

encdec_attention: base_pbs ${base_HDRS} ${encdec_attention_SRCS}
	g++ ${CFLAGS} -o $@ ${encdec_attention_SRCS} ${base_LIBS}
