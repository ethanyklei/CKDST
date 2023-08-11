# Speech Data Preprocessing
We use MUTS-C data to carry out our experiment. Then we will introduce how to preprocess the MUTS-C data.

First, you need download [MUST-C](https://ict.fbk.eu/must-c/). Then you should set `DATA_ROOT` to the root of MUST-C data.
We recommend setting the `REPO_ROOT` variable to avoid potential path errors.

When the above variables have been set, you can run the following command to automatically preprocess the MUST-C data.
```bash
bash prep_mustc_data.sh [TGT]
```
Where `[TGT]` is the target language (e.g. de).