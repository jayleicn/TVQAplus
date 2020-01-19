# Dependencies:
-- Python 2.7
-- PyTorch 1.2
-- Numpy

# Run sample:
$ cd tvqa_plus_eval_release
$ bash eval_sample.sh
Expected output:
"""
...Start TVQAplus eval...

Start evaluation...
QA Acc. 1.0
Grd. mAP 0.999057237396
Temp. mIoU0.999337089824
ASA 0.999337089824
"""

# Run on your own prediction:
Please prepare the prediction file in the same format as the `data/sample_submission.json'.
`data/sample_submission.json' is essentially the output of `load_tvqa_plus_annotation' function
when its `anno_path' parameter is `data/tvqa_plus_val.json'. Thus, it should be helpful to look
at the `load_tvqa_plus_annotation' function to get an idea of the exact format of the submiision
file. After the preparation, run
$ python eval_tvqa_plus.py --pred_path ${path_to_your_prediction_file}
