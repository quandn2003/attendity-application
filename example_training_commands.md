# Training Examples with New Arguments

## Basic Training with Default Settings
```bash
python3 ai/models/train_lfw.py --data_dir data --num_epochs 20
```
- Test evaluation: every 10 epochs
- Checkpoint saving: every 10 epochs

## Fast Training for Development (Test More Frequently)
```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --num_epochs 20 \
    --epoch_size 50 \
    --test_every 5 \
    --save_every 5
```
- Test evaluation: every 5 epochs
- Checkpoint saving: every 5 epochs

## Production Training (Test Less Frequently, Save More Often)
```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --num_epochs 50 \
    --epoch_size 1000 \
    --test_every 20 \
    --save_every 5
```
- Test evaluation: every 20 epochs
- Checkpoint saving: every 5 epochs

## Kaggle P100 Optimized (4-hour budget)
```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --num_epochs 20 \
    --epoch_size 50 \
    --batch_size 120 \
    --learning_rate 0.01 \
    --people_per_batch 60 \
    --images_per_person 20 \
    --pretrained vggface2 \
    --num_workers 2 \
    --test_every 10 \
    --save_every 5
```
- Test evaluation: every 10 epochs (2 times total)
- Checkpoint saving: every 5 epochs (4 times total)

## Quick Debugging (Frequent Testing and Saving)
```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --num_epochs 10 \
    --epoch_size 20 \
    --test_every 2 \
    --save_every 2
```
- Test evaluation: every 2 epochs
- Checkpoint saving: every 2 epochs

## Long Training (Minimal Testing, Regular Saving)
```bash
python3 ai/models/train_lfw.py \
    --data_dir data \
    --num_epochs 100 \
    --epoch_size 500 \
    --test_every 25 \
    --save_every 10
```
- Test evaluation: every 25 epochs (4 times total)
- Checkpoint saving: every 10 epochs (10 times total) 