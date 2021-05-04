def get_config(train, test, model):
    if train == test:
        return "[{}_trained_on_{}_tested_on_{}]\n".format(model, train, test) + \
            "TrainingName = {}_{}\n".format(models_name[model], datasets_name[train]) + \
            "DatasetName = {}\n\n".format(datasets_name[train])
    return "[{}_trained_on_{}_tested_on_{}_filtered_with_{}]\n".format(model, train, test, train) + \
        "TrainingName = {}_{}\n".format(models_name[model], datasets_name[train]) + \
        "DatasetName = {}_filtered_with_{}\n".format(datasets_name[test], datasets_name[train]) + \
        "[{}_trained_on_{}_tested_on_{}_filtered_with_{}]\n".format(model, test, test, train) + \
        "TrainingName = {}_{}\n".format(models_name[model], datasets_name[test]) + \
        "DatasetName = {}_filtered_with_{}\n".format(datasets_name[test], datasets_name[train])


datasets = ["onto", "bbn", "choi", "figer"]
datasets_name = {"onto": "OntoNotes",
                 "bbn": "BBN",
                 "choi": "Choi",
                 "figer": "FIGER"}

models = ["bert_ft_0", "bert_ft_2", "adapter_2", "adapter_16"]
models_name = {"bert_ft_0": "bertFeatureExtraction",
               "bert_ft_2": "bertFT2",
               "adapter_2": "adapters2",
               "adapter_16": "adapter16"}


with open("test.ini", "w") as test_file:
    test_file.write("[DEFAULT]\n" + \
                    "TrainingName = DEFAULT\n" + \
                    "DatasetName    = DEFAULT\n\n" + \
                    "PerformanceFile = results/performances/\n" + \
                    "PredictionFile  = results/predictions/\n" + \
                    "AvgStdFile      = results/avgs_stds/\n\n" + \
                    "DevOrTest = both\n\n\n") 


if __name__ == "__main__":
    for train in datasets:
        for test in datasets:
            for model in models:
                with open("test.ini", "a") as test_file:
                    test_file.write(get_config(train, test, model))
