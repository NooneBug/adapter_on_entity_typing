#!/bin/zsh

# BBN native
# list="bert_ft_2_trained_on_bbn_tested_on_bbn adapter_2_trained_on_bbn_tested_on_bbn adapter_16_trained_on_bbn_tested_on_bbn"

# Choi Native
# list="bert_ft_2_trained_on_choi_tested_on_choi adapter_2_trained_on_choi_tested_on_choi adapter_16_trained_on_choi_tested_on_choi"

# FIGER native
# list="bert_ft_2_trained_on_figer_tested_on_figer adapter_2_trained_on_figer_tested_on_figer adapter_16_trained_on_figer_tested_on_figer"

# Ontonotes native
# list="bert_ft_2_trained_on_onto_tested_on_onto adapter_2_trained_on_onto_tested_on_onto adapter_16_trained_on_onto_tested_on_onto"

# ---------------------------------------

# # BBN on onto filtered with BBN
# list="bert_ft_2_trained_on_bbn_tested_on_onto_filtered_with_bbn \
      # adapter_2_trained_on_bbn_tested_on_onto_filtered_with_bbn \
      # adapter_16_trained_on_bbn_tested_on_onto_filtered_with_bbn"

# # onto on onto filtered with BBN
# list="bert_ft_2_trained_on_onto_tested_on_onto_filtered_with_bbn \
#       adapter_2_trained_on_onto_tested_on_onto_filtered_with_bbn \
#       adapter_16_trained_on_onto_tested_on_onto_filtered_with_bbn"

# ---------------------------------------

# # BBN on FIGER filtered with BBN
# list="bert_ft_2_trained_on_bbn_tested_on_figer_filtered_with_bbn \
#       adapter_2_trained_on_bbn_tested_on_figer_filtered_with_bbn \
#       adapter_16_trained_on_bbn_tested_on_figer_filtered_with_bbn"

# # FIGER on FIGER filtered with BBN
# list="bert_ft_2_trained_on_figer_tested_on_figer_filtered_with_bbn \
#       adapter_2_trained_on_figer_tested_on_figer_filtered_with_bbn \
#       adapter_16_trained_on_figer_tested_on_figer_filtered_with_bbn"

# ---------------------------------------

# BBN on choi filtered with bbn
# list="bert_ft_2_trained_on_bbn_tested_on_choi_filtered_with_bbn \
#       adapter_2_trained_on_bbn_tested_on_choi_filtered_with_bbn \
#       adapter_16_trained_on_bbn_tested_on_choi_filtered_with_bbn"

# # Choi on choi filtered with bbn
# list="bert_ft_2_trained_on_choi_tested_on_choi_filtered_with_bbn \
#       adapter_2_trained_on_choi_tested_on_choi_filtered_with_bbn \
#       adapter_16_trained_on_choi_tested_on_choi_filtered_with_bbn"

# ---------------------------------------

# # Onto on bbn filtered with onto
# list="bert_ft_2_trained_on_onto_tested_on_bbn_filtered_with_onto \
#       adapter_2_trained_on_onto_tested_on_bbn_filtered_with_onto \
#       adapter_16_trained_on_onto_tested_on_bbn_filtered_with_onto"

# # BBN  on bbn filtered with onto
# list="bert_ft_2_trained_on_bbn_tested_on_bbn_filtered_with_onto \
      # adapter_2_trained_on_bbn_tested_on_bbn_filtered_with_onto \
      # adapter_16_trained_on_bbn_tested_on_bbn_filtered_with_onto"

# ---------------------------------------

# # Onto on figer filtered with onto
# list="bert_ft_2_trained_on_onto_tested_on_figer_filtered_with_onto \
      # adapter_2_trained_on_onto_tested_on_figer_filtered_with_onto \
      # adapter_16_trained_on_onto_tested_on_figer_filtered_with_onto"

# # FIGER  on figer filtered with onto
# list="bert_ft_2_trained_on_figer_tested_on_figer_filtered_with_onto \
      # adapter_2_trained_on_figer_tested_on_figer_filtered_with_onto \
      # adapter_16_trained_on_figer_tested_on_figer_filtered_with_onto"

# ---------------------------------------

# # Onto on choi filtered with onto
# list="bert_ft_2_trained_on_onto_tested_on_choi_filtered_with_onto \
      # adapter_2_trained_on_onto_tested_on_choi_filtered_with_onto \
      # adapter_16_trained_on_onto_tested_on_choi_filtered_with_onto"

# # choi  on choi filtered with onto
# list="bert_ft_2_trained_on_choi_tested_on_choi_filtered_with_onto \
      # adapter_2_trained_on_choi_tested_on_choi_filtered_with_onto \
      # adapter_16_trained_on_choi_tested_on_choi_filtered_with_onto"

# ---------------------------------------

# # FIGER on bbn filtered with figer
# list="bert_ft_2_trained_on_figer_tested_on_bbn_filtered_with_figer \
      # adapter_2_trained_on_figer_tested_on_bbn_filtered_with_figer \
      # adapter_16_trained_on_figer_tested_on_bbn_filtered_with_figer"

# # BBN  on bbn filtered with figer
# list="bert_ft_2_trained_on_bbn_tested_on_bbn_filtered_with_figer \
#       adapter_2_trained_on_bbn_tested_on_bbn_filtered_with_figer \
#       adapter_16_trained_on_bbn_tested_on_bbn_filtered_with_figer"

# ---------------------------------------

# # FIGER on onto filtered with figer
# list="bert_ft_2_trained_on_figer_tested_on_onto_filtered_with_figer \
      # adapter_2_trained_on_figer_tested_on_onto_filtered_with_figer \
      # adapter_16_trained_on_figer_tested_on_onto_filtered_with_figer"

# # onto  on onto filtered with figer
list="bert_ft_2_trained_on_onto_tested_on_onto_filtered_with_figer \
      adapter_2_trained_on_onto_tested_on_onto_filtered_with_figer \
      adapter_16_trained_on_onto_tested_on_onto_filtered_with_figer"

# ---------------------------------------

# # FIGER on choi filtered with figer
# list="bert_ft_2_trained_on_figer_tested_on_choi_filtered_with_figer \
#       adapter_2_trained_on_figer_tested_on_choi_filtered_with_figer \
#       adapter_16_trained_on_figer_tested_on_choi_filtered_with_figer"

# # Choi on choi filtered with figer
# list="bert_ft_2_trained_on_choi_tested_on_choi_filtered_with_figer \
#       adapter_2_trained_on_choi_tested_on_choi_filtered_with_figer \
#       adapter_16_trained_on_choi_tested_on_choi_filtered_with_figer"

# ---------------------------------------

# # Choi on bbn filtered with choi
# list="bert_ft_2_trained_on_choi_tested_on_bbn_filtered_with_choi \
      # adapter_2_trained_on_choi_tested_on_bbn_filtered_with_choi \
      # adapter_16_trained_on_choi_tested_on_bbn_filtered_with_choi"


# # BBN  on bbn filtered with choi
# list="bert_ft_2_trained_on_bbn_tested_on_bbn_filtered_with_choi \
#       adapter_2_trained_on_bbn_tested_on_bbn_filtered_with_choi \
#       adapter_16_trained_on_bbn_tested_on_bbn_filtered_with_choi"

# ---------------------------------------

# # Choi on onto filtered with choi
# list="bert_ft_2_trained_on_choi_tested_on_onto_filtered_with_choi \
#       adapter_2_trained_on_choi_tested_on_onto_filtered_with_choi \
#       adapter_16_trained_on_choi_tested_on_onto_filtered_with_choi"

# # onto on onto filtered with choi
# list="bert_ft_2_trained_on_onto_tested_on_onto_filtered_with_choi \
      # adapter_2_trained_on_onto_tested_on_onto_filtered_with_choi \
      # adapter_16_trained_on_onto_tested_on_onto_filtered_with_choi"

# ---------------------------------------

# # Choi on FIGER filtered with choi
# list="bert_ft_2_trained_on_choi_tested_on_figer_filtered_with_choi \
#       adapter_2_trained_on_choi_tested_on_figer_filtered_with_choi \
#       adapter_16_trained_on_choi_tested_on_figer_filtered_with_choi"

# # FIGER on FIGER filtered with choi
# list="bert_ft_2_trained_on_figer_tested_on_figer_filtered_with_choi \
#       adapter_2_trained_on_figer_tested_on_figer_filtered_with_choi \
#       adapter_16_trained_on_figer_tested_on_figer_filtered_with_choi"

# ---------------------------------------



for name in $list; do
  python test.py $name
done
