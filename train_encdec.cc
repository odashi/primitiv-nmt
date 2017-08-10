#include <iostream>
#include <string>
#include <vector>

#include "messages.pb.h"
#include "sampler.h"
#include "utils.h"

using namespace std;

int main(int argc, char *argv[]) {
  ::check_args(argc, argv, {
      "(int) Embedding size",
      "(int) Hidden size",
      "(file/in) Train corpus file",
      "(file/in) Development corpus file",
      "(file/in) Source vocabulary file",
      "(file/in) Target vocabulary file",
  });

  return 0;
}
