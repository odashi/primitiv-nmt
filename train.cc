#include <iostream>

#include "mymt.pb.h"
#include "sampler.h"
#include "utils.h"

using namespace std;

#include <cstdio>

int main(int argc, char *argv[]) {
  mymt::proto::Corpus train_corpus, dev_corpus;

  cout << "Loading train corpus ..." << endl;
  ::load_proto("model/ende.train.corpus", train_corpus);
  cout << "Loading dev corpus ..." << endl;
  ::load_proto("model/ende.newstest2012.corpus", dev_corpus);

  ::RandomBatchSampler train_sampler(train_corpus, 16, 0);
  ::RandomBatchSampler dev_sampler(dev_corpus, 16, 0);

  unsigned n = 0;
  map<unsigned, unsigned> freqs;
  for (unsigned i = 0; train_sampler.has_next(); ++i) {
    ::Batch batch = train_sampler.next();
    n += batch.source[0].size();
    ++freqs[batch.source[0].size()];
    cout << "batch " << i << ':' << endl;
    cout << "  source:" << endl;
    for (const auto &words : batch.source) {
      cout << "   ";
      for (unsigned id : words) printf(" %5d", id);
      cout << endl;
    }
    cout << "  target:" << endl;
    for (const auto &words : batch.target) {
      cout << "   ";
      for (unsigned id : words) printf(" %5d", id);
      cout << endl;
    }
  }
  cout << n << endl;
  for (const auto &kv : freqs) cout << kv.first << ' ' << kv.second << endl;

  return 0;
}
