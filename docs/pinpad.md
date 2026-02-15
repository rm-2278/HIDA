configs.yaml:
Task can be chosen from 'pinpad_three' to 'pinpad_eight'
Adjust the steps, eval_every, eval_eps, log_every as needed here.
Don't forget to adjust wandb settings here as well.

To run:
```
python hieros/train.py --configs pinpad
```



Note: evaluation takes quite a long time.



hierarchy:
```
python hieros/train.py --configs pinpad hierarchy_decrease --task=pinpad_three --batch_size=8 --batch_length=32
```