import sys
import os
import urllib
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing, svm
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from openpyxl.workbook import Workbook

def main():
    st.title("Анализ данных & Машинное обучение")
    st.subheader("Выберите файл с данными для анализа")

    def select_file(folder_path = './datasets'):
        filename = os.listdir(folder_path)
        seltd_flname= st.selectbox("Выбереите файл с данными для анализа",filename)
        return os.path.join(folder_path,seltd_flname)
    fname = select_file()
    st.info("Вы выбрали {}".format(fname))

    data = pd.read_csv(fname)

    def save_exel(df,SName):
        writer = pd.ExcelWriter('Results.xlsx')
        df.to_excel(writer,SName )

        writer.save()
    if st.checkbox("Показать данные"):
        num = st.slider("Колличество строк для отображеия ",5,data.shape[0])
        st.dataframe(data.head(num))

    if st.checkbox("Заголовки столбцов"):
        st.write(data.columns)

    if st.checkbox("Показать размерность"):

        data_dm = st.radio("Размерность по",("Строкам","Столбцам"))
        if data_dm == "Столбцам":
            st.write("Число столбцов: "+str(data.shape[1]))
        elif data_dm == "Строкам":
            st.write("Число строк: "+str(data.shape[0]))




    if st.checkbox("Типы данных"):
        st.write(data.dtypes)

    if st.checkbox("Статистика по значениям"):
        st.write(data.describe().T)

    if st.checkbox("Рассмотреть отдельные столбцы"):
        all_data = data.columns.tolist()
        sltd_columns = st.multiselect("Select",all_data)
        new_data = data[sltd_columns]
        st.dataframe(new_data)
    st.header("Визуализация данных")
    if st.checkbox("Построить корреляционную матрицу"):
        plt.figure(figsize=(10,10))
        plt.title('Correlation between different fearures')

        st.write(sns.heatmap(data.corr(),vmax=1, square=True, annot=True, cmap='gray_r'))
        st.pyplot()

    st.subheader("Построение графиков")
    all_clummn_names=data.columns.tolist()
    plot_type = st.selectbox("Выберите тип графика",["area","bar","line","hist","box","kde"])
    selected_columns = st.multiselect("Выберите столбцы для построения графика",all_clummn_names)

    if st.button("Построить график"):
        st.success("Построение графика {} для {}".format(plot_type,selected_columns))
        if plot_type == 'area':
            plot_data = data[selected_columns]
            st.area_chart(plot_data)
        elif plot_type == 'bar':
            plot_data = data[selected_columns]
            st.bar_chart(plot_data)
        elif plot_type == 'line':
            plot_data = data[selected_columns]
            st.line_chart(plot_data)
        elif plot_type:
            cust_data = data[selected_columns].plot(kind=plot_type)
            st.write(cust_data)
            st.pyplot()

    if st.checkbox ("Круговая диаграмма"):
        all_clummn_names = data.columns.tolist()
        if st.button("Построить круговую диаграмму"):
            st.success("Поостроение круговой диаграммы")
            st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    def regression_errors_values(testY, predict, n_predict):
        st.write("Средняя абсолютная ошибка:", mean_absolute_error(testY, predict))
        st.write("Средняя квадратичная ошибка:", mean_squared_error(testY, predict))
        st.write("Средняя абсолютная ошибка (c нормализацией):", mean_absolute_error(testY, n_predict))
        st.write("Средняя квадратичная ошибка (c нормализацией):", mean_squared_error(testY, n_predict))

    def regression_plot_show( trainX , testY , predict, n_predict, reg_type ,target_label):
        xx = [i for i in range(trainX.shape[0])]
        plt.figure()
        plt.plot(xx[0:testY.size], testY[0:testY.size], 'o', color='r', label='y')
        plt.plot(xx[0:testY.size], predict[0:testY.size], color='b', linewidth=2, label='predicted y')
        plt.plot(xx[0:testY.size], n_predict[0:testY.size], color='k', linewidth=2, label='predicted y with normalize')
        plt.ylabel(target_label)
        plt.xlabel('Line number in dataset')
        plt.legend(loc=4)
        plt.title(reg_type)

        st.pyplot(plt.show())



    def model_linear_Regression(trainX, trainY,testX,testY):
        model = LinearRegression(normalize=False)
        model.fit(trainX, trainY)

        predict = model.predict(testX)
        return predict

    def model_ridge_Regression(trainX, trainY, testX,testY):
        model = Ridge(normalize=True)
        model.fit(trainX, trainY)

        predict = model.predict(testX)
        return predict


    def model_lasso(trainX, trainY, testX,testY):
        model = Lasso(normalize=False)
        model.fit(trainX, trainY)

        predict = model.predict(testX)
        return predict


    def model_random_forest(trainX, trainY,  testX,testY):
        model = RandomForestRegressor(criterion="mae", bootstrap=True)
        model.fit(trainX, trainY)
        predict = model.predict(testX)
        return predict


    if st.checkbox("PCA проекция"):
        scaler = StandardScaler()
        pca = decomposition.PCA(n_components=2)
        X_reduced = pca.fit_transform(scaler.fit_transform(data))
        plt.figure(figsize=(6, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                    edgecolor='none', alpha=0.7, s=40,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))
        plt.title('Feautures PCA projection')
        st.pyplot(plt.show())

    if st.checkbox("TSNE проекция"):
        scaler = StandardScaler()
        tsne = TSNE(random_state=17)
        tsne_representation = tsne.fit_transform(scaler.fit_transform(data))
        plt.figure(figsize=(6, 6))
        plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
                    edgecolor='none', alpha=0.7, s=40,
                    cmap=plt.cm.get_cmap('nipy_spectral', 10))
        plt.title('Feautures T-sne projection ')
        st.pyplot(plt.show())
    st.subheader ("Применение методов машинного обучения")
    if st.checkbox("Выбрать тип решаемой задачи"):
        problem_type = st.radio("Выберите тип задачи", ("Регрессия","Кластеризация", "Классификация"))

        if problem_type == "Регрессия":
            st.write("Выберите столбец у для задачи регрессии")
            selected_y = st.multiselect( "",all_clummn_names)
            reg_type = st.radio("Выберите алгоритм решения", ("Все","Линейная регрессия", "Ridge", "Lasso", "Случайный лес"))
            test_size_slider = st.slider("Выберите размер тестовой выборки %", 1, 100)
            if st.checkbox("Выполнить"):
                st.success("Столбец {} выбран успешно".format(selected_y))

                train, test = train_test_split(data, test_size=test_size_slider/100)
                st.write(test_size_slider/100)
                trainX = np.array(train.drop(selected_y, 1))
                trainY = np.array(train[selected_y])
                testX = np.array(test.drop(selected_y, 1))
                testY = np.array(test[selected_y])

                st.write(selected_y)

                st.write(trainY.reshape(trainY.shape[0]))




                if reg_type == "Все":
                    reg_type = "Линейная регрессия"
                    n_predictions = model_linear_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                    predictions = model_linear_Regression(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)


                    reg_type = "Ridge"
                    n_predictions = model_ridge_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                    predictions = model_ridge_Regression(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)

                    reg_type = "Случайный лес"
                    n_predictions = model_random_forest(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                    predictions = model_random_forest(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)


                    reg_type = "Lasso"
                    n_predictions = model_lasso(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                    predictions = model_lasso(trainX, trainY, testX, testY)

                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)

                    reg_type=''


                    if st.checkbox("Cохранить результаты регрессии"):
                        cols = []
                        for i in all_clummn_names:
                            cols.append(str(i))
                        cols.remove(selected_y[-1])
                        cols.append(str(selected_y[-1]) + " ( Selected y )")

                        df1 = pd.DataFrame(np.concatenate((trainX,trainY),axis=1),columns=cols)
                        st.write(df1)
                        cols.pop()
                        cols.append('Predictions')
                        cols.append(str(selected_y[-1])+" ( Real y value )")

                        st.write(selected_y[-1])

                        st.write(predictions)

                        #res = np.concatenate((testX, predictions), axis=1)
                        predictions= np.reshape(predictions,(predictions.shape[-1],1))
                        st.write(predictions.shape)
                        st.write(predictions)
                        df2 = pd.DataFrame(np.concatenate((np.concatenate((testX,predictions),axis=1),testY),axis=1),columns=cols)
                        st.write(df2)

                        writer = pd.ExcelWriter ('Линейная регрессия.xlsx')
                        df1.to_excel(writer, 'Обучающая выборка')
                        df2.to_excel(writer, 'Тестовая ваборка')
                        writer.save()



                if reg_type == "Линейная регрессия":
                    n_predictions = model_linear_Regression(preprocessing.normalize(trainX), trainY, preprocessing.normalize(testX), testY)
                    predictions = model_linear_Regression(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)


                    if st.checkbox("Cохранить результаты регрессии"):
                        cols = []
                        for i in all_clummn_names:
                            cols.append(str(i))
                        cols.remove(selected_y[-1])
                        cols.append(str(selected_y[-1]) + " ( Selected y )")

                        df1 = pd.DataFrame(np.concatenate((trainX,trainY),axis=1),columns=cols)
                        st.write(df1)
                        cols.pop()
                        cols.append('Predictions')
                        cols.append(str(selected_y[-1])+" ( Real y value )")


                        predictions= np.reshape(predictions,(predictions.shape[0],1))


                        df2 = pd.DataFrame(np.concatenate((np.concatenate((testX,predictions),axis=1),testY),axis=1),columns=cols)
                        st.write(df2)

                        writer = pd.ExcelWriter ('Линейная регрессия.xlsx')
                        df1.to_excel(writer, 'Обучающая выборка')
                        df2.to_excel(writer, 'Тестовая ваборка')
                        writer.save()


                if reg_type == "Ridge":
                    n_predictions = model_ridge_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX), testY)
                    predictions = model_ridge_Regression(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)

                if reg_type == "Lasso":

                    n_predictions = model_lasso(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                    predictions = model_lasso(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)

                if reg_type == "Случайный лес":
                    n_predictions = model_random_forest(preprocessing.normalize(trainX), trainY, preprocessing.normalize(testX), testY)
                    predictions = model_random_forest(trainX, trainY, testX, testY)
                    regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                    regression_errors_values(testY, predictions, n_predictions)




        elif problem_type == "Кластеризация":
            n_clusters = st.number_input("Введите количество кластеров:",3)
            dataset = data.copy()

            scaler = StandardScaler()
            X = scaler.fit_transform(dataset)
            km = KMeans(n_clusters=n_clusters)

            # fit & predict clusters
            dataset['cluster'] = km.fit_predict(X)
            st.write(dataset['cluster'])
            st.write(dataset)

            if st.checkbox("PCA"):
                pca = decomposition.PCA(n_components=2)
                X_reduced = pca.fit_transform(X)
                plt.figure(figsize=(6, 6))
                plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                            edgecolor='none', alpha=0.7, s=40, c=dataset['cluster'],
                            cmap=plt
