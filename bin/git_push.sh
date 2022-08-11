# cloud synchronization between multiple machines causes git hooks to not be executable
export PYTHONPATH=$PYTHONPATH:$(pwd)/ML_for_Battery_Design/external/BayesFlow 
pre-commit install -t pre-push
git push