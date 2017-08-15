protobuf_generate_cpp(mymt_proto_SRCS mymt_proto_HDRS mymt.proto)

set(mymt_all_HDRS
  ${mymt_proto_HDRS}
  encoder_decoder.h
  lstm.h
  sampler.h
  nmt_utils.h
  utils.h
  vocabulary.h
)

function(mymt_compile name)
  add_executable(${name} ${mymt_all_HDRS} ${name}.cc ${mymt_proto_SRCS})
  target_link_libraries(${name} ${PRIMITIV_LIBRARIES} ${PROTOBUF_LIBRARIES})
endfunction()

mymt_compile(make_vocab)
mymt_compile(dump_vocab)
mymt_compile(make_corpus)
mymt_compile(dump_corpus)
mymt_compile(train_encdec)
mymt_compile(resume_encdec)
