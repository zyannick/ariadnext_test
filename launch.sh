for loss_type in   'cosface'  'sphereface' 'arcface' 'no'
do
    python3 main.py -loss_type=$loss_type -num_epochs=20
done