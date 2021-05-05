#!/bin/zsh

# BBN native
# list="bert_ft_2_trained_on_bbn_tested_on_bbn adapter_2_trained_on_bbn_tested_on_bbn adapter_16_trained_on_bbn_tested_on_bbn"

# Choi Native
# list="bert_ft_2_trained_on_choi_tested_on_choi adapter_2_trained_on_choi_tested_on_choi adapter_16_trained_on_choi_tested_on_choi"

# FIGER native
# list="bert_ft_2_trained_on_figer_tested_on_figer adapter_2_trained_on_figer_tested_on_figer adapter_16_trained_on_figer_tested_on_figer"

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

# # BBN on FIGER filtered with BBN
# list="bert_ft_2_trained_on_bbn_tested_on_figer_filtered_with_bbn \
#       adapter_2_trained_on_bbn_tested_on_figer_filtered_with_bbn \
#       adapter_16_trained_on_bbn_tested_on_figer_filtered_with_bbn"

# # FIGER on FIGER filtered with BBN
# list="bert_ft_2_trained_on_figer_tested_on_figer_filtered_with_bbn \
#       adapter_2_trained_on_figer_tested_on_figer_filtered_with_bbn \
#       adapter_16_trained_on_figer_tested_on_figer_filtered_with_bbn"

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

# BBN on choi filtered with bbn
# list="bert_ft_2_trained_on_bbn_tested_on_choi_filtered_with_bbn \
#       adapter_2_trained_on_bbn_tested_on_choi_filtered_with_bbn \
#       adapter_16_trained_on_bbn_tested_on_choi_filtered_with_bbn"

# # Choi on choi filtered with bbn
# list="bert_ft_2_trained_on_choi_tested_on_choi_filtered_with_bbn \
#       adapter_2_trained_on_choi_tested_on_choi_filtered_with_bbn \
#       adapter_16_trained_on_choi_tested_on_choi_filtered_with_bbn"

# ---------------------------------------

# # FIGER on choi filtered with figer
# list="bert_ft_2_trained_on_figer_tested_on_choi_filtered_with_figer \
#       adapter_2_trained_on_figer_tested_on_choi_filtered_with_figer \
#       adapter_16_trained_on_figer_tested_on_choi_filtered_with_figer"

# # Choi on choi filtered with figer
list="bert_ft_2_trained_on_choi_tested_on_choi_filtered_with_figer \
      adapter_2_trained_on_choi_tested_on_choi_filtered_with_figer \
      adapter_16_trained_on_choi_tested_on_choi_filtered_with_figer"

# ---------------------------------------

for name in $list; do
  python test.py $name
done
