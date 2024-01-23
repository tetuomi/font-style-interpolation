#!/bin/sh

# echo 'FANnet evaluation'
# pipenv run python eval_fannet_weight_interpolation.py
# pipenv run python eval_fannet_letter_recognition.py


CHAR='A'
DM_MODEL_PATH='./weight/log41_fannet_retrain_step_final.pth'
echo 'Noise evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Condition evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Image evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR


CHAR='B'
DM_MODEL_PATH='./weight/log42_fannet_retrain_step_final.pth'
echo 'Noise evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Condition evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Image evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR


CHAR='C'
DM_MODEL_PATH='./weight/log43_fannet_retrain_step_final.pth'
echo 'Noise evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Noise' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Condition evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Condition' -model_path $DM_MODEL_PATH -char $CHAR

echo 'Image evaluation'
pipenv run python eval_dm_weight_interpolation.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR
pipenv run python eval_dm_letter_recognition.py -approach 'Image' -model_path $DM_MODEL_PATH -char $CHAR
