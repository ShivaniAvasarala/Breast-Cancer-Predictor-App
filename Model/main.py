import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scaling Data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Train Model
    model = LogisticRegression()
    model.fit(X_train, y_train)


    # Test Model
    y_pred = model.predict(X_test)
    print('Accuracy of the model: ', accuracy_score(y_test, y_pred))
    print("Classification report: " , classification_report(y_test, y_pred))


    return model, scaler


def get_clean_data():
    pd.set_option('display.max_columns', None)
    data = pd.read_csv("Data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis']= data['diagnosis'].map({'M': 1, 'B': 0})


    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open('Model/model.pkl', 'wb')as f:
        pickle.dump(model, f)
    with open('Model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)



if __name__ == '__main__':
    main()