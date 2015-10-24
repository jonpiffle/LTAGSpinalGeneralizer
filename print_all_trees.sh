#!/bin/sh

FILENAME='uncompressed_trees_u_rules.json'

# Create new file with opening brace
`rm $FILENAME; printf '[' > $FILENAME;`

# Generate trees one at a time
for ((i=0; i <= 24; i++)); do
    echo jython -J-XX:+UseConcMarkSweepGC -J-Xmx1g print_generalized_trees.py $i
    `jython -J-XX:+UseConcMarkSweepGC -J-Xmx1g print_generalized_trees.py $i $FILENAME`
done

# Remove trailing comma
`sed -i '' '$ s/.$//' $FILENAME`

# Close opening brace
`printf ']' >> $FILENAME`