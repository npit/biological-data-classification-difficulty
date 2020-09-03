import arff
import pandas as pd
def read_arff_data(filepath):
    # with open (filepath) as f:
    #     decoder=arff.ArffDecoder()
    #     import ipdb; ipdb.set_trace()
    #     datadictionary=decoder.decode(f,encode_nominal=True,return_type=arff.LOD)
    #     data=datadictionary['data']
    # import ipdb; ipdb.set_trace()

    with open (filepath) as f:
        data = arff.load(f)['data']
    full_df = pd.DataFrame(data)

    # encode nominal class
    class_col = full_df.columns[-1]
    for class_idx, class_name in enumerate(set(full_df[class_col])):
        idx = full_df[full_df.columns[-1]] == class_name
        full_df.loc[idx, class_col] = class_idx


    return full_df