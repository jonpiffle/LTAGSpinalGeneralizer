#!/bin/sh

FILENAME='output/propbank_args.txt'

# Create new file with opening brace
`rm $FILENAME; touch $FILENAME;`

# Generate trees one at a time
for ((i=0; i <= 24; i++)); do
    echo jython -J-XX:+UseConcMarkSweepGC -J-Xmx1g print_generalized_trees.py $i
    `jython -J-XX:+UseConcMarkSweepGC -J-Xmx1g print_generalized_trees.py $i $FILENAME print_args`
done
