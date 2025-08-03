  Compression Ratio Analysis Results

  TinyStories Tokenizer (10K vocabulary):
  - On TinyStories text: 2.62 bytes/token
  - On OpenWebText text: 2.61 bytes/token

  OpenWebText Tokenizer (32K vocabulary):
  - On TinyStories text: 2.61 bytes/token
  - On OpenWebText text: 2.65 bytes/token

  Key Findings:

  1. Domain Matching Effect: Each tokenizer performs slightly better on its training domain, though the difference is small
   (0.4-1.5% improvement).
  2. Vocabulary Size Impact: Interestingly, the larger 32K OpenWebText tokenizer doesn't consistently achieve better
  compression ratios than the smaller 10K TinyStories tokenizer. This suggests that domain-specific training data can be
  more important than vocabulary size for compression efficiency.
  3. Overall Performance: Both tokenizers achieve similar compression ratios (~2.6 bytes/token), indicating that BPE
  tokenization is quite robust across different text domains.

  The results show that while vocabulary size and domain matching do matter, the differences are relatively modest, with
  all compression ratios falling in the 2.5-2.8 bytes/token range.