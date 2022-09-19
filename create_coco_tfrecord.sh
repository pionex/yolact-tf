python3 create_coco_tfrecord.py --logtostderr \
      --include_masks="True" \
      --image_dir="./dataset/images" \
      --object_annotations_file="./dataset/images/train.json" \
      --output_file_prefix="./dataset/train/train" \
      --num_shards=88

