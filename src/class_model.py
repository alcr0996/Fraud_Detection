import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score , recall_score, classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier, plot_tree, plot_importance

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})


class DataCleaner(object):

    def __init__(self, data_frame):
        """
        Instantiate with a one row dataframe (from HTML cleaning class)
        ex:
            X = DataCleaner(data_frame)
            X_vals = X.get_array_of_values()
        """
        self.event_name = None
        self.single_df = data_frame
        self.total_vars = 0
        self.engineer_features()
        if self.total_vars < 1:
            self.whole_df = self.single_df
            self.total_vars = 1
        else:
            self.whole_df.append(self.single_df)
            self.total_vars += 1
        self.X = self.single_df.values[0]
    
    def set_new_data_point(self, data_frame):
        self.single_df = data_frame
        self.whole_df.append(self.single_df)
        self.total_vars += 1

    def engineer_features(self):
        self.single_df.fillna({"has_analytics" : 0, "has_header" : 0, "has_logo" : 0, "delivery_method" : 0}, inplace=True)
        self.single_df["event_duration"] = self.single_df["event_end"] - self.single_df["event_start"]
        self.single_df["event_turnover"] = self.single_df["event_published"] - self.single_df["event_created"]
        self.single_df["user_turnover"] = self.single_df["event_created"] - self.single_df["user_created"]
        self.single_df['num_ticket_types'] = [len(i) for i in self.single_df['ticket_types']]
        self.single_df['total_payouts'] = [len(i) for i in self.single_df['previous_payouts']]
        self.single_df["payout_specified"] = [0 if method == '' else 1 for method in self.single_df["payout_type"]]
        self.single_df["is_channel_0"] = [1 if channel == 0 else 0 for channel in self.single_df["channels"]]
        self.single_df["is_delivery_0"] = [1 if method == 0 else 0 for method in self.single_df["delivery_method"]]
        self.set_rounded_ticket_averages()
        self.set_country_matching_event()
        self.event_name = self.single_df["name"].values
        self.drop_extra_cols()
        self.single_df.fillna(0, inplace=True)
        
    def set_rounded_ticket_averages(self):
        averages = [] 
        for i in self.single_df['ticket_types']: 
            avg_price = []
            for y in i: 
                price = y['cost']
                avg_price.append(price)
                avg = np.mean(avg_price)
            averages.append(avg)
        self.rounded_averages = [round(i, 2) for i in averages] 
        self.single_df['average_ticket_price'] = self.rounded_averages

    def set_country_matching_event(self):
        for i in range(len(self.single_df)):
            v_cond1 = self.single_df.loc[i, "venue_country"] == None
            v_cond2 = self.single_df.loc[i, "venue_country"] == ""
            c_cond1 = self.single_df.loc[i, "country"] == None
            c_cond2 = self.single_df.loc[i, "country"] == ""
            if v_cond1 or v_cond2:
                self.single_df.loc[i, "venue_country"] = self.single_df.loc[i, "country"]
            if c_cond1 or c_cond2:
                self.single_df.loc[i, "country"] = self.single_df.loc[i, "venue_country"]
        self.single_df["country_matching_event"] = self.single_df["country"] == self.single_df["venue_country"]

    def drop_extra_cols(self):
        self.single_df.drop(["org_facebook", "org_twitter", "channels", "venue_name", "email_domain",
                "previous_payouts", "ticket_types", "payee_name", "payout_type",
                "country", "venue_country", "has_analytics", "has_header", "delivery_method",
                "has_logo", "event_created", "event_end", "event_published",
                "event_start", "user_created", "approx_payout_date", "currency",
                "fb_published", "gts", "name_length", "num_order", "num_payouts",
                "listed", "object_id", "sale_duration", "sale_duration2", "venue_address",
                "venue_latitude", "venue_longitude", "venue_state", "description", "name", "org_desc",
                "org_name"], axis=1, inplace=True)

def algorithm_pipeline(X_train, X_test, y_train, y_test, 
                       model, param_grid, cv=6):
    RSC = RandomizedSearchCV(
        estimator=model,
        param_grid=param_grid, 
        cv=cv, 
        n_jobs=-1,
        verbose=2,
        n_iter = 20
    )
    model = RSC.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    return model, y_pred

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

if __name__ == "__main__":
    df = pd.read_json('../data/data.json')
    df['fraud'] = np.where(df['acct_type'].str.contains('fraud'), 1, 0)
    df.drop('acct_type', axis=1, inplace=True)
    
    clean = DataCleaner(df)

    y = df.pop('fraud')
    X = df

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)
    
    model = XGBClassifier(
        max_depth = 10,
        learning_rate = 0.58,
        n_estimators = 40,
        reg_alpha = 0.6,
        reg_lambda = 0.7
        )
    # param_grid = {
    #     'max_depth' : [10],
    #     'learning_rate' : [0.58],
    #     'n_estimators' : [40],
    #     'reg_alpha' : [0.6],
    #     'reg_lambda' : [0.7]
    # }
    
    # model, y_pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model,
    #                 param_grid, cv=6)

    # print(np.sqrt(-model.best_score_))
    # print(model.best_params_)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('Classification Report')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    
    plot_confusion_matrix(y_test, y_pred, title='Confusion matrix')
    plt.savefig('../images/confusion_matrix.png')
    plt.show()

    print('AUROC Score')
    print(roc_auc_score(y_test, y_pred))
    
    # plot_importance(model)
    # plt.tight_layout()
    # plt.show()

    