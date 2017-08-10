CFLAGS=-std=c++11 -Ofast -Wall -Werror

base_HDRS=utils.h vocabulary.h
base_PROTOS=mymt.proto
base_PB_CCS=mymt.pb.cc
base_LIBS=-lprimitiv -lprotobuf

make_vocab_SRCS=make_vocab.cc ${base_PB_CCS}
dump_vocab_SRCS=dump_vocab.cc ${base_PB_CCS}
make_corpus_SRCS=make_corpus.cc ${base_PB_CCS}
dump_corpus_SRCS=dump_corpus.cc ${base_PB_CCS}
train_encdec_SRCS=train_encdec.cc ${base_PB_CCS}

all: make_vocab dump_vocab make_corpus dump_corpus train_encdec

base_pbs: ${base_PROTOS}
	protoc --cpp_out=. ${base_PROTOS}

make_vocab: base_pbs ${base_HDRS} ${make_vocab_SRCS}
	g++ ${CFLAGS} -o $@ ${make_vocab_SRCS} ${base_LIBS}

dump_vocab: base_pbs ${base_HDRS} ${dump_vocab_SRCS}
	g++ ${CFLAGS} -o $@ ${dump_vocab_SRCS} ${base_LIBS}

make_corpus: base_pbs ${base_HDRS} ${make_corpus_SRCS}
	g++ ${CFLAGS} -o $@ ${make_corpus_SRCS} ${base_LIBS}

dump_corpus: base_pbs ${base_HDRS} ${dump_corpus_SRCS}
	g++ ${CFLAGS} -o $@ ${dump_corpus_SRCS} ${base_LIBS}

train_encdec: base_pbs ${base_HDRS} ${train_encdec_SRCS}
	g++ ${CFLAGS} -o $@ ${train_encdec_SRCS} ${base_LIBS}
