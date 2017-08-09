protobuf_generate_cpp(
  primitiv_nmt_proto_SRCS primitiv_nmt_proto_HDRS primitiv_nmt.proto)

set(primitiv_nmt_all_HDRS
  ${primitiv_nmt_proto_HDRS}
  affine.h
  attention.h
  encoder_decoder.h
  lstm.h
  sampler.h
  nmt_utils.h
  utils.h
  vocabulary.h
)

function(primitiv_nmt_compile name)
  add_executable(${name} ${primitiv_nmt_all_HDRS} ${name}.cc ${primitiv_nmt_proto_SRCS})
  target_link_libraries(${name} ${PRIMITIV_LIBRARIES} ${PROTOBUF_LIBRARIES})
endfunction()

primitiv_nmt_compile(make_vocab)
primitiv_nmt_compile(dump_vocab)
primitiv_nmt_compile(make_corpus)
primitiv_nmt_compile(dump_corpus)

primitiv_nmt_compile(train)
primitiv_nmt_compile(resume)
primitiv_nmt_compile(translate)
primitiv_nmt_compile(translate_ensemble)
