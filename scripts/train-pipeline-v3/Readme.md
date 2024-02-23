General overview of notebooks:

1. Load the data, preprocess and split into train and test on image-level. Creates image-level CSVs and images folder with images that have been revieved.
2. Take the CSVs from notebook 1 and convert them to COCO json files to be compatible with tiling script. Creates image-level COCO JSON files.
3. Take image-level tiling JSON files and convert the dataset into tiles. Creates tile-level COCO JSONs and tile-level images folder.
4. Subsample the generated tiles. There will be lots of empty tiles that will be discarded. Creates subsamples tile-level COCO JSON files and images folder. Converts these JSONs to yolo-compatible .txt files, guides in creating the yolo-compatible dataset directory structure. Helps in creating the yolo config.yaml file.
5. Trains and evaluates the yolo model on subsampled tiles dataset from notebook 4.