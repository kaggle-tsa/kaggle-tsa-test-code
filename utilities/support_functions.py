import pandas as pd

def prediction_dataframe(ids, predictions, zone):
    n = predictions.shape[0]
    column_names = ['Id', 'Probability']
    df = pd.DataFrame(columns = column_names, index = range(n))

    for i in range(n):
        id = ids[i] + '_' + zone
        score = predictions[i][1]
        df['Id'][i] = id
        df['Probability'][i] = score

    return df


def default_prediction_dataframe(ids, default_value, zone):
    n = ids.size
    column_names = ['Id', 'Probability']
    df = pd.DataFrame(columns = column_names, index = range(n))

    for i in range(n):
        id = ids[i] + '_' + zone
        score = default_value
        df['Id'][i] = id
        df['Probability'][i] = score

    return df

def write_scores_to_csv(submission_ids, new_scores, dir):

    submission_ids.set_index('Id', inplace=True)
    new_scores.set_index('Id', inplace=True)
    submission_ids.update(new_scores)
    submission_ids = submission_ids.reindex(columns=['Probability'])
    submission_ids.to_csv(dir)
