import json
from fbprophet.serialize import model_from_json
import pylab as plt
import pandas as pd

time = {'start_time': '2018-12-02', 'end_time': '2021-12-01'}

future = pd.DataFrame({'ds': pd.date_range(start=time['start_time'], end=time['end_time'])})

with open('./serialized_model.json', 'r') as f:
    model = model_from_json(json.load(f))

predict = model.predict(future)

predict['yhat']