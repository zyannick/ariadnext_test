for loss_type in   'cosface'  'sphereface' 'arcface' 'no'
do
    python3 main.py -loss_type=$loss_type
done