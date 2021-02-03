#!/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names
mv breast-cancer-wisconsin.data data/raw
mv breast-cancer-wisconsin.names data/raw

