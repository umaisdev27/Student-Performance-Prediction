import dill
with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = dill.load(f)

# check what columns it expects
print(preprocessor.feature_names_in_)