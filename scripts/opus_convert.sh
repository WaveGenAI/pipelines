#!/bin/bash

# check if the target directory is provided
if [ "$#" -ne 1 ]; then
    echo "Use: $0 /path/to/target/directory"
    exit 1
fi

TARGET_DIR="$1"

# check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "$TARGET_DIR NOT FOUND"
    exit 1
fi

# convert mp3 to opus
find "$TARGET_DIR" -iname "*.mp3" -type f | parallel -I% --max-args 1 \
    "ffmpeg -i % -strict -2 -c:a opus -b:a 128K -vbr on -map_metadata 0 -compression_level 10 -y %.opus; touch -r % %.opus; rm -vf %"

# remove mp3 files
find "$TARGET_DIR" -iname "*.mp3" -type f -exec rm -vf {} \;

# update metadata.csv
awk -F, 'BEGIN {OFS=","} {if (NR>1) $1 = gensub(/\.mp3$/, ".opus", "g", $1); print}' "$TARGET_DIR/metadata.csv" > "$TARGET_DIR/metadata1.csv"

# rename metadata1.csv to metadata.csv
mv -vf "$TARGET_DIR/metadata1.csv" "$TARGET_DIR/metadata.csv"
