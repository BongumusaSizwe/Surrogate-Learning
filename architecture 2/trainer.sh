# Script to test qdn network

# Training for different games learning rates

echo "Training different learning rates"

for i in 0.1 0.01 0.001 0.0001 0.00001 0.000001;
do;
	python train_atari.py --lr $i;
done;
ech "Done Training learning rates"

##Training for different batch sizes
echo "Starting different Batch Training"
for i in 4 8 16 32 64 128;
do;
	python train_atari.py --batch-size $i;
done;
echo "Done Training different batches"

#Training
