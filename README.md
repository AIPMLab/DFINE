

This code is from https://github.com/ShihuaHuang95/DEIM, The innovative code proposed in this article is located in.\engine\extre_module

# Train
```python
python train.py -c configs/deim_dfine/.yml --use-amp --seed=0
``` 

# Test
``` python
python train.py -c configs/deim_dfine/.yml --test-only -r model.pth

```
