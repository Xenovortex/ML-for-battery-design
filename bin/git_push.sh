# cloud synchronization between multiple machines causes git hooks to not be executable
export PYTHONPATH=$PYTHONPATH:$(pwd)/ML_for_Battery_Design/external/BayesFlow
export PYTHONPATH=$PYTHONPATH:$(pwd)/ML_for_Battery_Design/external/exchange_HSU_UniHD
pre-commit install -t pre-push
git push