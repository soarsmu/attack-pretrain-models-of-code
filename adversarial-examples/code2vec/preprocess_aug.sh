#!/usr/bin/env bash

PYTHON=python3
EXTRACTOR=JavaExtractor/extract.py
NUM_THREADS=4

for d in $1; do
    if [[ -d $d ]]; then
        # $f is a directory
		#echo "$d"
		for f in $d/*; do
			# run path extractor on both original and mutants
			echo $f
			${PYTHON} ${EXTRACTOR} --dir ${f} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > ${f}/augtempboth
			${PYTHON} preprocess_test_batch.py --test_data ${f}/augtempboth --max_contexts 200 --dict_file data/java14m/java14m --output_name ${f}/both
			rm ${f}/augtempboth
			
			#run path extractor on original file
			for origin in $f/src/*; do
				#echo  "${origin:(-13):(-5)}"
				if [ "${origin:(-13):(-5)}" != "_mutants" ]; then
					#echo $origin
					${PYTHON} ${EXTRACTOR} --file ${origin} --max_path_length 8 --max_path_width 2 --num_threads ${NUM_THREADS} --jar JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar > ${f}/augtemporigin
					${PYTHON} preprocess_test_batch.py --test_data ${f}/augtemporigin --max_contexts 200 --dict_file data/java14m/java14m --output_name ${f}/original
					rm ${f}/augtemporigin
				fi
			done
		done
	fi
done