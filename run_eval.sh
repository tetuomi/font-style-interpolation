#!/bin/sh

# echo 'FANnet evaluation'
# pipenv run python eval_fannet_weight_interpolation.py
# pipenv run python eval_fannet_letter_recognition.py

echo 'Noise evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Noise'
pipenv run python eval_dm_letter_recognition.py -approach 'Noise'

echo 'Condition evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Condition'
pipenv run python eval_dm_letter_recognition.py -approach 'Condition'

echo 'Image evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Image'
pipenv run python eval_dm_letter_recognition.py -approach 'Image'
