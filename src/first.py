import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score , recall_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.inspection import partial_dependence, plot_partial_dependence

# store keyword argument default values
tmpdefaults = XGBClassifier.predict_proba.__defaults__
# change default value of validate_features to False
XGBClassifier.predict_proba.__defaults__ = (None, False)

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 18})

    
def create_DataFrame(DataFrame):
    new_df = DataFrame.copy()
    ticket_averages(new_df)
    ticket_range(new_df)
    total_quantity_tickets(new_df)
    engineer_features(new_df)
    drop_cols(new_df)
    return new_df

def engineer_features(DataFrame):
    DataFrame.fillna({"delivery_method_is_0" : 0}, inplace=True)
    DataFrame["event_end_min_start"] = DataFrame["event_end"] - DataFrame["event_start"]
    DataFrame["event_start_min_published"] = DataFrame["event_start"] - DataFrame["event_published"]
    DataFrame["event_end_min_published"] = DataFrame["event_end"] - DataFrame["event_published"]
    DataFrame["event_published_min_created"] = DataFrame["event_published"] - DataFrame["event_created"]
    DataFrame["event_start_min_created"] = DataFrame["event_start"] - DataFrame["event_created"]
    DataFrame["event_end_min_created"] = DataFrame["event_end"] - DataFrame["event_created"]
    DataFrame["user_turnover"] = DataFrame["event_created"] - DataFrame["user_created"]
    DataFrame['num_ticket_types'] = [len(i) for i in DataFrame['ticket_types']]
    DataFrame['total_prev_payouts'] = [len(i) for i in DataFrame['previous_payouts']]
    DataFrame["payout_specified"] = [0 if method == '' else 1 for method in DataFrame["payout_type"]]
    DataFrame["channel_is_0"] = [1 if channel == 0 else 0 for channel in DataFrame["channels"]]
    DataFrame["channel_is_5"] = [1 if channel == 5 else 0 for channel in DataFrame["channels"]]
    DataFrame["channel_is_6"] = [1 if channel == 6 else 0 for channel in DataFrame["channels"]]
    DataFrame["delivery_method_0"] = [1 if method == 0 else 0 for method in DataFrame["delivery_method"]]
    DataFrame.fillna(0, inplace=True)
    
def ticket_averages(DataFrame):
    averages = [] 
    for row in DataFrame['ticket_types']: 
        avg_price = []
        for i in row: 
            price = i['cost']
            avg_price.append(price)
            avg = np.mean(avg_price)
        averages.append(avg)
    rounded_averages = [np.round(row, 2) for row in averages] 
    DataFrame['average_ticket_price'] = rounded_averages
    DataFrame.fillna(0, inplace=True)

def ticket_range(DataFrame):
    ranges = [] 
    for row in DataFrame['ticket_types']: 
        lowest_price = []
        max_price = []
        for i in row:
            low_price = i['cost']
            if lowest_price == []:
                lowest_price.append(low_price)
            else:
                if low_price < lowest_price[0]:
                    lowest_price.pop(0)
                    lowest_price.append(low_price)
        for i in row:
            high_price = i['cost']
            if max_price == []:
                max_price.append(high_price)
            else:
                if high_price > max_price[0]:
                    max_price.pop(0)
                    max_price.append(high_price)
        range_per_row = np.array((max_price)- np.array(lowest_price))
        if range_per_row.size < 1:
            ranges.append(0)
        else:
            ranges.append(np.round(int(range_per_row)))
    
    DataFrame['range_ticket_price'] = ranges

def total_quantity_tickets(DataFrame):
    totals = [] 
    for row in DataFrame['ticket_types']:
        row_totals = [] 
        for i in row: 
            dict_total = i['quantity_total'] - i['quantity_sold']
            row_totals.append(dict_total)
            sums = np.sum(row_totals)
        totals.append(sums)
    DataFrame['total_tickets_sold'] = totals

def drop_cols(DataFrame):
    DataFrame.drop(['acct_type', 'approx_payout_date', 'channels', 'country',
       'currency', 'delivery_method', 'description', 'email_domain',
       'event_created', 'event_end', 'event_published', 'event_start',
       'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo',
       'listed', 'name', 'name_length', 'num_order', 'num_payouts',
       'object_id', 'org_desc', 'org_facebook', 'org_name', 'org_twitter',
       'payee_name', 'payout_type', 'previous_payouts', 'sale_duration',
       'sale_duration2', 'ticket_types', 'user_created', 'venue_address',
       'venue_country', 'venue_latitude', 'venue_longitude', 'venue_name',
       'venue_state'], axis=1, inplace=True)

def algorithm_pipeline(X_train, X_test, y_train, y_test, 
                       model, param_grid, cv=6):
    RSC = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid, 
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
    
    new_df = create_DataFrame(df)

    y = new_df.pop('fraud')
    X = new_df

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y)
    X_test_full = X_test.copy()
    X_test_full['Fraud loss']=X_test_full['average_ticket_price']*X_test_full['total_tickets_sold']
    X_test.drop('total_tickets_sold', axis = 1, inplace = True)
    X_train.drop('total_tickets_sold', axis = 1, inplace = True)
    
    model = XGBClassifier(
        max_depth = 10,
        learning_rate = 0.58,
        n_estimators = 40,
        reg_alpha = 0.6,
        reg_lambda = 0.7,
        random_state=42
    )
    param_grid = {
        'max_depth' : [10],
        'learning_rate' : [0.58],
        'n_estimators' : [40],
        'reg_alpha' : [0.6],
        'reg_lambda' : [0.7]
    }
    
    model, y_pred = algorithm_pipeline(X_train, X_test, y_train, y_test, model,
                    param_grid, cv=6)

    print(np.sqrt(-model.best_score_))
    print(model.best_params_)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    print('Classification Report')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))

    print('AUROC Score')
    print(roc_auc_score(y_test, y_pred))
    # plot importance only works on model when not ran through randomizedsearchcv
    # plot_importance(model)
    # plt.tight_layout()
    # plt.savefig('../images/feature_important_all.png')
    # plt.show()

    plot_confusion_matrix(y_test, y_pred, title='Confusion matrix')

    # sns.pairplot(new_df)
    # plt.tight_layout()
    # plt.show()

    pickle.dump(model, open('../models/model_all_features.pkl', 'wb')) 

    col_list = new_df.columns
    for col in col_list:
        partial_dependency(model, X_test, new_df.columns, col)
    fpr, tpr, thresholds = roc_curve(y_test, y_probs[:, 1])
    plt.title('ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TRP')
    plt.plot(fpr, tpr, lw=5, label= 'ROC')

    # partial dependence plots
    feats = np.arange(0, 20, 1)

    for feat in feats:
        name = 'partial_dependency_feat_'+str(feat)+'.png'
        plot_partial_dependence(estimator=model, X=X_test, features=[feat], feature_names=X_test.columns.tolist(), n_cols=1)
        plt.savefig('../images/'+name)

    # reset default keyword argument values to original
    XGBClassifier.predict_proba.__defaults = tmpdefaults





    