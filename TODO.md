# TODO List

DEADLINE: 27/04

/datahdd/vmanuel/entity_typing_all_datasets/data/entity_typing_original_datasets/mapped_datasets

- [X] Manuel : generate filtered datasets with original labels
- [X] Manuel : generate results for bert_ft_2 for choi
- [X] Manuel : train and generate results for bert_ft_2 for figer
- [X] Manuel : generate results of from scratch model on filtered datasets (DOMAIN ADAPTATION):
  - [X] trained from scratch on figer, tested on:
    - [X]  BBN filtered with figer (BBN_into_FIGER)
    - [X]  ONTO filtered with figer (Ontonotes_into_FIGER)
    - [X]  CHOI filtered with figer (choi_into_FIGER)
  - [X] trained from scratch on choi, tested on:
    - [X]  BBN filtered with choi (BBN_into_choi)
    - [X]  ONTO filtered with choi (Ontonotes_into_choi)
    - [X]  FIGER filtered with choi (figer_into_choi)
- [X] Manuel : generate results of from scratch model on filtered datasets:
  - [X] trained from scratch on figer, tested on:
    - [X] FIGER filtered with BBN (figer_into_BBN)
    - [X] FIGER filtered with onto (figer_into_Ontonotes)
    - [X] FIGER filtered with choi (figer_into_choi)
  - [X] trained from scratch on choi, tested on:
    - [X] choi filtered with BBN (choi_into_BBN)
    - [X] choi filtered with onto (choi_into_Ontonotes)
    - [X] choi filtered with FIGER (choi_into_figer)

- [ ] Manuel: train feature extractor on choi for 4 times
- [ ] Manuel: train feature extractor on figer for 4 times
- [ ] Manuel: compute results of FE on choi and figer
- [ ] Manuel: compute dataset stats:
  - [ ] datasets lengths/composition
  - [ ] datasets intersection

- [X] Federico : computare le performance come sopra
- [X] Federico : disegnare grafici performances (micro e macro_examples):
  - [X] grafico diviso in 4 sottografici:
    - [X] sottografico BBN: barchart con 4 bin (BBN test, BBN test mapped on Onto, BBN test mapped on figer, BBN test mapped on choi)
      - [X] ogni barchart ha 3 (4?) barre (bert_ft_2_bbn, adapters_2_bbn, adapters_16_bbn, Onoe(?))   

- [ ] compute p-values on micro and macro
- [ ] write results on latex
- [ ] Take a vacation for a YEAR
