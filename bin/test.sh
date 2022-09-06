export PYTHONPATH=$PYTHONPATH:$(pwd)/ML_for_Battery_Design/external/BayesFlow
export PYTHONPATH=$PYTHONPATH:$(pwd)/ML_for_Battery_Design/external/exchange_HSU_UniHD
pytest --cov=ML_for_Battery_Design/src --cov-fail-under=100 -vv