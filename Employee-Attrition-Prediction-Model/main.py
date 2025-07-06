from AIProjekt import load_data, encode_categorical, plot_initial_visualizations, feature_engineering, clean_data, train_model,save_artifacts
from gui import gui
def main():
    file_path = 'IBM_HR_Employee_Attrition.csv'

    df = load_data(file_path)
    print('.\n')
    # explore_data(df)
    df = encode_categorical(df)
    print('.\n')
    # plot_initial_visualizations(df)
    df = feature_engineering(df)
    print('.\n')
    df = clean_data(df)
    print('.\n')
    model, y_test, y_pred, accuracy = train_model(df)
    print('.\n')
    save_artifacts(model, df, y_test, y_pred, accuracy)
    print('.\n')
    gui(model, df)

if __name__ == "__main__":
    main()