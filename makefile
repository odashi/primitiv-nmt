CFLAGS=-std=c++11

base_HDRS=utils.h vocab.h
base_PROTOS=messages.proto
base_PB_CCS=messages.pb.cc
base_LIBS=-lprimitiv -lprotobuf

make_vocab_SRCS=make_vocab.cc ${base_PB_CCS}
dump_vocab_SRCS=dump_vocab.cc ${base_PB_CCS}
encdec_attention_SRCS=encdec_attention.cc ${base_PB_CCS}

all:

base_pbs: ${base_PROTOS}
	protoc --cpp_out=. ${base_PROTOS}

make_vocab: base_pbs ${base_HDRS} ${make_vocab_SRCS}
	g++ ${CFLAGS} -o $@ ${make_vocab_SRCS} ${base_LIBS}

dump_vocab: base_pbs ${base_HDRS} ${dump_vocab_SRCS}
	g++ ${CFLAGS} -o $@ ${dump_vocab_SRCS} ${base_LIBS}

encdec_attention: base_pbs ${base_HDRS} ${encdec_attention_SRCS}
	g++ ${CFLAGS} -o $@ ${encdec_attention_SRCS} ${base_LIBS}
