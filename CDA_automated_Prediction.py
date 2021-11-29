import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.preprocessing import MinMaxScaler

# import darts
# import darts
# import matplotlib.pyplot as plt
# import darts
# from darts.models import RNNModel
# import seaborn
# import matplotlib
import holidays


# ohne Aktion Datenverlauf
#

def load_data():
    return


def train_test_split_(X, y):
    """
    Splits the overall dataset in train and test(validattion data)
    :param X: Independet variables (covariates) such as promotions of the specified yoghurt.
    :param y: Dependent variable -> sales of the specified yoghurt
    :return: Split of train and test data
    """
    # Train Test Split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 8, shuffle=False)
    train_until = 285
    validate_from = 286
    X_train = X.loc[:train_until]
    X_test = X.loc[validate_from:]
    y_train = y.loc[:train_until]
    y_test = y.loc[validate_from:]

    return X_train, X_test, y_train, y_test


def linear_model_(X_train, y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    :param X_train: training covariates
    :param y_train: training sales
    :param X_test: testing covariates
    :return: a prediction with the same length as the test data
    """
    # Initialize Model
    lr_model = linear_model.LinearRegression()

    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def support_vector_machines(X_train, y_train, X_test):
    """
    Support vector machines
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def linear_model_lasso(X_train,y_train, X_test):
    """
    Trains a linear Regression Model and predicts Values for the specified Timestamps.
    :param X_train: training covariates
    :param y_train: training sales
    :param X_test: testing covariates
    :return: a prediction with the same length as the test data
    """
    # Initialize Model
    #lr_model = linear_model.Lasso(alpha=0.1)
    lr_model = linear_model.Lasso(alpha=0.5)


    # Train Linear Regression Model
    lr_model.fit(X_train, y_train)

    # Predict Datapoints
    y_pred = lr_model.predict(X_test)

    return y_pred


def random_forest_regressor(X_train,y_train, X_test, max_depth, random_state):
    """
    Random Forest Regressor
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    return y_pred

def neural_network(X_train, y_train, X_test, n_epochs):
    """
    Neural Network Keras from Tensorflow
    :param X_train:
    :param y_train:
    :param X_test:
    :return:
    """
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize Model
    model = Sequential()
    model.add(Dense(4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train Linear Regression Model
    model.fit(X_train, y_train, epochs=n_epochs)

    # Predict Datapoints
    y_pred_arr = model.predict(X_test)
    df_pred = pd.DataFrame(y_pred_arr, columns=['y_pred'])
    y_pred = df_pred['y_pred'].values
    return y_pred


def get_lag():
    return


def evaluate_predictions(y_test, y_pred_lin, y_pred_RFR, y_pred_SVM, y_pred_NN):
    """
    Generate Evaluation Factors for the predictions
    :param y_test: test sales
    :param y_pred: prediction sales
    :return: RMSE = Root Mean Squared Error, MAPE = Mean Absolute Percentage Error, R2-Score
    """
    # RMSE = np.sqrt(((y_test - y_pred) ** 2).mean())
    # print("Root Mean Squared Error: ", RMSE)
    MAPE_lin = np.mean(np.abs((y_test - y_pred_lin) / y_test)) * 100
    MAPE_RFR = np.mean(np.abs((y_test - y_pred_RFR) / y_test)) * 100
    MAPE_SVM = np.mean(np.abs((y_test - y_pred_SVM) / y_test)) * 100
    MAPE_NN = np.mean(np.abs((y_test.values - y_pred_NN) / y_test.values)) * 100

    # print("Mean Absolute Percentage Error: ", MAPE)
    # R2_score = r2_score(y_test, y_pred)
    return MAPE_lin, MAPE_RFR, MAPE_SVM, MAPE_NN


def plot_results(y_test,y_pred_lin, y_pred_RFR, y_pred_SVM,y_pred_NN,y_pred_LAREG,X_test,  plot_title, folder):
    """
    Plot results
    :param y_test: test sales
    :param y_pred: prediction sales
    :return:
    """

    # X_test = X_test[np.abs(X_test)< .1]= 0

    fig = plt.figure(figsize=[10, 7.5])
    #plt.plot(y_test.values, color='blue', label="Sales (Validation Values)")
    plt.plot(y_pred_lin, color='red', label="Sales (Prediction Linear Model)")
    plt.plot(y_pred_RFR, color='green', label="Sales (Prediction Random Forest Regressor)")
    plt.plot(y_pred_NN, color='purple', label="Sales (Prediction Random Forest Regressor2)")
    plt.plot(y_pred_SVM, color='yellow', label="Sales (Prediction Support Vector Machines)")

    plt.plot(y_pred_LAREG, color='black', label="Sales (Prediction Lasso Regression)")
    # plt.plot(X_test["promo_01"].values*np.mean(y_test)/3, marker = "x", label = "promo_01", linestyle = 'None')
    # plt.plot(X_test["promo_02"].values*np.mean(y_test)/3, marker = "x", label = "promo_02", linestyle =  'None')
    # plt.plot(X_test["promo_03"].values*np.mean(y_test)/3, marker = "x", label = "promo_03", linestyle =  'None')
    # plt.plot(X_test["promo_04"].values*np.mean(y_test)/3, marker = "x", label = "promo_04", linestyle = 'None')
    # plt.plot(X_test["promo_05"].values*np.mean(y_test)/3, marker = "x", label = "promo_05", linestyle = 'None')
    plt.xlabel('Validation Set Number')
    plt.ylabel('Sales')
    plt.title(plot_title)
    plt.grid()
    plt.ylim([200, max(y_test) * 1.3])
    #plt.ylim([5000, 22000])
    plt.legend()
    # plt.show()
    directory = os.path.join(
        r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\Prediction_Plots", folder)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig(os.path.join(directory,plot_title.replace('/','')+'.jpeg'))

    return


if __name__ == "__main__":

    basepath_exp = r'C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\data_hs21'
    basepath_test = r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\test_data"

    #folder = "X_Promo_single_akumuliert_ohne_lag"
    #folder = "X_Promo_all_single_akumuliert_ohne_lag"
    #folder = "X_Promo_all_ohne_lag"
    #folder = "X_Promo_single_ohne_lag"
    lst_folders = ["X_Promo_single_akumuliert_ohne_lag", "X_Promo_all_single_akumuliert_ohne_lag","X_Promo_all_ohne_lag",  "X_Promo_single_ohne_lag"]
    for folder in lst_folders:

        Dateiname = folder + ".csv"

        range_numbers = range(0, 19)
        # range_numbers = [0]
        X_Promo_all = pd.DataFrame()
        X_Promo_all_single_akumuliert = pd.DataFrame()


        for n in range_numbers:
            dateiname_exp = r'article_' + str(n) + '.csv'
            dateiname_test = r'article_' + str(n) +'_test.csv'
            df_exp = pd.read_csv(os.path.join(basepath_exp, dateiname_exp))
            df_test = pd.read_csv(os.path.join(basepath_test, dateiname_test))
            df_test = df_test.rename(columns={"artikel_name": "article_name", "promo_1": "promo_01", "promo_2": "promo_02","promo_3": "promo_03", "promo_4": "promo_04", "promo_5": "promo_05"})
            df = pd.concat([df_exp, df_test], axis = 0)
            df.index = range(len(df))

            Promo = df[["promo_01", "promo_02", "promo_03", "promo_04", "promo_05"]]
            Promo.columns += "_" + str(n)

            Promo_akumuliert = (df["promo_01"] + df["promo_02"] + df["promo_03"] + df["promo_04"] + df["promo_05"])
            Promo_akumuliert.name = "promo_" + str(n)

            X_Promo_all = pd.concat([X_Promo_all, Promo], axis=1)
            X_Promo_all_single_akumuliert = pd.concat([X_Promo_all_single_akumuliert, Promo_akumuliert], axis=1)
        print(X_Promo_all)
        print(X_Promo_all_single_akumuliert)


        for n in range_numbers:
            if n == 8:
                dateiname_exp = r'article_' + str(n) + '.csv'
                dateiname_test = r'article_' + str(n) +'_test.csv'
                df_exp = pd.read_csv(os.path.join(basepath_exp, dateiname_exp))
                df_test = pd.read_csv(os.path.join(basepath_test, dateiname_test))
                df_test = df_test.rename(columns={"artikel_name": "article_name","promo_1": "promo_01","promo_2": "promo_02", "promo_3": "promo_03", "promo_4": "promo_04", "promo_5": "promo_05"})
                df = pd.concat([df_exp, df_test], axis=0)
                df.index = range(len(df))

                X_temproal = df[["year", "month", "week"]]
                X_Promo_single = df[["promo_01", "promo_02", "promo_03", "promo_04", "promo_05"]]
                X_Promo_single_accumulated = (
                            df["promo_01"] + df["promo_02"] + df["promo_03"] + df["promo_04"] + df["promo_05"])
                X_Promo_single_accumulated.name = "promo"
                # X_Promo_single_accumulated.columns = ["promo"]
                # X = pd.concat([X_temproal, X_Promo_all, X_lag], axis = 1)

                if folder == "X_Promo_single_akumuliert_ohne_lag":
                    X = pd.concat([X_temproal, X_Promo_single_accumulated], axis=1)
                elif folder == "X_Promo_all_single_akumuliert_ohne_lag":
                    X = pd.concat([X_temproal, X_Promo_all_single_akumuliert], axis=1)
                elif folder == "X_Promo_all_ohne_lag":
                    X = pd.concat([X_temproal, X_Promo_all], axis=1)
                elif folder == "X_Promo_single_ohne_lag":
                    X = pd.concat([X_temproal, X_Promo_single], axis=1)

                # df = pd.read_csv(os.path.join(basepath, r'article_2.csv'))
                y = df["sales"]
                #        y = df["sales"]
                # X = df[["year", "yrwk_start", "yrwk_end", "month", "week", "promo_01", "promo_02", "promo_03", "promo_04", "promo_05"]]
                # X = df[["year", "month", "week", "promo_01", "promo_02", "promo_03", "promo_04", "promo_05"]]

                # Train Test Split
                X_train, X_test, y_train, y_test = train_test_split_(X, y)

                # Linear Model
                y_pred_lin = linear_model_(X_train, y_train, X_test)

                # Random Forest Regressor
                y_pred_RFR = random_forest_regressor(X_train, y_train, X_test, max_depth=2, random_state=0)

                # Support Vector Machines
                y_pred_SVM = support_vector_machines(X_train, y_train, X_test)

                # Lasso Regression
                y_pred_LAREG = linear_model_lasso(X_train, y_train, X_test)

                # Neural Network Keras from Tensorflow
                # y_pred_NN = y_pred_SVM * np.NAN
                #n_epochs = 100
                #y_pred_NN = neural_network(X_train, y_train, X_test, n_epochs)
                y_pred_RFR2 = random_forest_regressor(X_train, y_train, X_test, max_depth=4, random_state=0)

                # Evaluation Factors
                #MAPE_lin, MAPE_RFR, MAPE_SVM, MAPE_NN = evaluate_predictions(y_test, y_pred_lin, y_pred_RFR, y_pred_SVM,
                #                                                             y_pred_NN)

                #dict_results = {"Artikel_Name": df["article_name"].tolist()[0], "MAPE_lin": MAPE_lin, "MAPE_RFR": MAPE_RFR,
                #                "MAPE_SVM": MAPE_SVM, "MAPE_NN": MAPE_NN}
                # Plot Results
                plot_results(y_test,y_pred_lin, y_pred_RFR, y_pred_SVM,y_pred_RFR2,y_pred_LAREG,X_test, df["article_name"].tolist()[0],
                             folder)
                df_prediction = pd.DataFrame(data = {"article_name": df_test["article_name"], "y_pred_lin":y_pred_lin,"y_pred_RFR":y_pred_RFR, "y_pred_RFR2":y_pred_RFR2, "y_pred_SVM":y_pred_SVM, "y_pred_LAREG": y_pred_LAREG}, index = df_test.index)

                directory_pred = os.path.join(r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\Prediction_Data",folder)
                if not os.path.exists(directory_pred):
                    os.makedirs(directory_pred)

                df_prediction.to_csv(os.path.join(directory_pred, dateiname_test))
                print(df_prediction)
        # df_eval_single
        #df_eval_single = pd.DataFrame(data=dict_results, index=[str(n)])
        #df_evaluation = pd.concat([df_evaluation, df_eval_single], axis=0)
        #print(df_evaluation)
    #print(df_evaluation)



    # df_evaluation.to_csv(os.path.join(r"C:\Users\pascs\OneDrive\Desktop\Master_Data_Science\Customer Data Analytics\CDA_Project\Evaluation Matrix", Dateiname))


