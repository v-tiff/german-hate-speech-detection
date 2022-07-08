import numpy as np
import pandas as pd
from sklearn import metrics

from src.utils import read_survey_CSV

SURVEY_CSV_PATH = "../../survey_results.csv"

if __name__ == '__main__':
    df = pd.read_csv(SURVEY_CSV_PATH, sep=",")
    # Use data starting from 12th column
    df = df.iloc[:, 12:]
    df = df.transpose(copy=True).reset_index()
    df.rename(columns={'index': 'text'}, inplace=True)
    df.replace(to_replace=['🟢 NO HATE', '🔴 HATE'], value=[0, 1], inplace=True)

    survey_df = read_survey_CSV()

    column_list = ["tp", "fp", "fn", "tn", "accuracy", "f1_score"]
    survey_results_df = pd.DataFrame(columns=column_list)

    for i in range(1, len(df.columns)):
        single_result_df = pd.concat([survey_df['input_text'], survey_df['target_text'], df.iloc[:, i]], axis=1,
                                     keys=['text', 'y_true', 'y_pred'])

        y_true, y_pred = single_result_df['y_true'], single_result_df['y_pred']

        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        accuracy = metrics.accuracy_score(y_true, y_pred)
        f1_score = metrics.f1_score(y_true, y_pred)

        single_result_row = pd.DataFrame(data=np.array([[tp, fp, fn, tn, accuracy, f1_score]]), columns=column_list)
        survey_results_df = pd.concat([survey_results_df, single_result_row],
                                      ignore_index=True)

    survey_results_df.to_csv('../../survey_metrics.csv')
