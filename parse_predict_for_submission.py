import numpy as np
import pandas as pd

predictions = np.load('../predictions.npy')


prediction_dict = {'id_code':[], 'diagnosis':[]}

diagnosis = np.zeros((len(predictions)), dtype='int32')

for i in range (0,len(predictions)):
    diagnosis[i] = int(np.where(predictions[i] == np.amax(predictions[i]))[0])

prediction_dict['diagnosis']=diagnosis

test_df= pd.read_csv('../test.csv')

test_values = test_df.values
id_code = [value[0] for value in test_values]

prediction_dict['id_code'] = id_code

prediction = pd.DataFrame(prediction_dict)

#finally, create the csv file for submission
prediction.to_csv('dretinophaty_predictions.csv')
