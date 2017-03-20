## TODO


## LUNA
* generate blobs: test_seg_scan.py  
* probabilities for blobs (fpred): test_fpred_scan.py
* stats over segmentation blobs: evaluate_luna_seg_scan.py
* stats over fpred: evaluate_luna_fpred_scan.py


## DSB

* generate blobs: test_seg_scan_dsb.py
* fpred: test_fpred_scan_dsb.py
* plot rois as in the final data iterator: plot_dsb_roi.py
* train classifier: train_class_dsb.py 



## List of patients
1.3.6.1.4.1.14519.5.2.1.6279.6001.943403138251347598519939390311 (nodule on the border, bad quality)
1.3.6.1.4.1.14519.5.2.1.6279.6001.287966244644280690737019247886 (biggest nodule 32 mm)
b8bb02d229361a623a4dc57aa0e5c485 (has 2 series of data)
08528b8817429d12b7ce2bf444d264f9 (half of the lung)
6a145c28d3b722643f547dfcbdf379ae (half of the lung)
5fe048f36bd2da6bdb63d8ff3c4022cd (half of the lung)
51fbac477a3639f983904fc4d42b8c15 (the highest candidate is probably not a nodule)

## Random
ssh -X -p 10001 ikorshun@localhost
scp -P 10001 /mnt/sda3/data/kaggle-lung/stage1_sample_submission.csv  ikorshun@localhost://mnt/storage/data/dsb3/
