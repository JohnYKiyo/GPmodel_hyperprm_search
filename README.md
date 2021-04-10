# GPlearn

## Usage
```
python GPRegression.py --scale 1.0 --alpha 0.1
```

または、
```
from GPRegression import load_data, train, test
X_train, X_test, Y_train, Y_test = load_data()
gpr = train(args.scale, args.alpha)
RMSE = test(gpr)
hoge
```
