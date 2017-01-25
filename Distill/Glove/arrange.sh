python rankinput_source_target.py $1 "$1"1
python permute_source_target.py 128 "$1"1 "$1"2
rm "$1"1
mv "$1"2 $1
