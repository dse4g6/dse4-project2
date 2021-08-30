import pickle
import pandas as pd

# load data
model = pickle.load(open("socialtime_model.pickle", 'rb'))
scaler = pickle.load(open("socialtime_scaler.pickle", 'rb'))
encoder = pickle.load(open("socialtime_encoder.pickle", 'rb'))

def pred_pipeline(inp, model=model, sc=scaler, enc=encoder):
  columns = ['sex', 'age', 'mobile os', 'have girl/boyfriend']
  names = enc.get_feature_names([columns[0]] + columns[2:])
  s = pd.DataFrame({c: [inp[i]] for i, c in enumerate(columns)})
  encoded = enc.transform(s[[columns[0]] + columns[2:]])
  encoded = pd.DataFrame(encoded.toarray(), columns=names)
  x_data = pd.concat([s['age'], encoded], axis=1)
  inp_norm = sc.transform(x_data)
  pred = model.predict(inp_norm)
  return pred
