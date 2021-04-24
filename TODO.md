# TODO List

DEADLINE: 27/04

/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/mapped_datasets

- [X] Manuel : generate filtered datasets with original labels
- [X] Manuel : generate results for bert_ft_2 for choi
- [ ] Manuel : train and generate results for bert_ft_2 for figer
- [ ] Manuel : generate results of from scratch model on filtered datasets (DOMAIN ADAPTATION):
  - [ ] trained from scratch on figer, tested on:
    - [ ]  BBN filtered with figer (BBN_into_FIGER)
    - [ ]  ONTO filtered with figer (Ontonotes_into_FIGER)
    - [ ]  CHOI filtered with figer (choi_into_FIGER)
  - [ ] trained from scratch on choi, tested on:
    - [ ]  BBN filtered with choi (BBN_into_choi)
    - [ ]  ONTO filtered with choi (Ontonotes_into_choi)
    - [ ]  FIGER filtered with choi (figer_into_choi)
- [ ] Manuel : generate results of from scratch model on filtered datasets:
  - [ ] trained from scratch on figer, tested on:
    - [ ] FIGER filtered with BBN (figer_into_BBN)
    - [ ] FIGER filtered with onto (figer_into_Ontonotes)
    - [ ] FIGER filtered with choi (figer_into_choi)
  - [ ] trained from scratch on choi, tested on:
    - [ ] choi filtered with BBN (choi_into_BBN)
    - [ ] choi filtered with onto (choi_into_Ontonotes)
    - [ ] choi filtered with FIGER (choi_into_figer)
- [ ] Manuel : compute p-values on micro and macro
- [ ] Manuel : write results on latex

- [ ] Federico : computare le performance come sopra
- [ ] Federico : disegnare grafici performances (micro e macro_examples):
  - [ ] grafico diviso in 4 sottografici:
    - [ ] sottografico BBN: barchart con 4 bin (BBN test, BBN test mapped on Onto, BBN test mapped on figer, BBN test mapped on choi)
      - [ ] ogni barchart ha 3 (4?) barre (bert_ft_2_bbn, adapters_2_bbn, adapters_16_bbn, Onoe(?))   

- [ ] Take a vacation for a YEAR
