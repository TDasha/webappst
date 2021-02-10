
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import base64
import io


def main():

    def get_table_download_link2(df1,df2,frase):

        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        df1.to_excel(writer, 'Обучающая выборка')
        df2.to_excel(writer, 'Тестовая ваборка')

        writer.save()
        xlsx_io.seek(0)

        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        data = base64.b64encode(xlsx_io.read()).decode("utf-8")
        href = f'<a href="data:{media_type};base64,{data}" download="RegressionResults.xlsx" >{frase}</a> (right-click and save)'
        st.markdown(href, unsafe_allow_html=True)


    def save_train_test_data_xlsx(trainX, trainY,testX,testY,predictions):
        cols = []
        for i in all_clummn_names:
            cols.append(str(i))
        cols.remove(selected_y[-1])
        cols.append(str(selected_y[-1]) + " ( Selected y )")

        df1 = pd.DataFrame(np.concatenate((trainX, trainY), axis=1), columns=cols)
        st.text("Обучающая выборка")
        st.write(df1)
        cols.pop()
        cols.append('Predictions')
        cols.append(str(selected_y[-1]) + " ( Real y value )")

        predictions = np.reshape(predictions, (predictions.shape[0], 1))

        df2 = pd.DataFrame(np.concatenate((np.concatenate((testX, predictions), axis=1), testY), axis=1),
                           columns=cols)
        st.text("Тестовая выборка")
        st.write(df2)
        get_table_download_link2(df1, df2,"Сохранить результаты xlsx File")


    def save_exel(df,frase):
        xlsx_io = io.BytesIO()
        writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
        df.to_excel(writer, 'Результаты кластеризации')
        writer.save()
        xlsx_io.seek(0)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        data = base64.b64encode(xlsx_io.read()).decode("utf-8")
        href = f'<a href="data:{media_type};base64,{data}" download="ClusteringResults.xlsx" >{frase}</a> (right-click and save)'
        st.markdown(href, unsafe_allow_html=True)


    def get_file():
        file = st.file_uploader(" Нажмите browse files, чтобы загрузить файл в формате .csv", type="csv")
        show_file = st.empty()

        if not file:
            show_file.info(" Файл не загружен. Загрузите файл для анализа в формате .csv")
            return
        else: show_file.info(" Загрузка файла выполнена успешно")
        return file

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

    st.title("Анализ данных & Машинное обучение")
    st.subheader("Загрузите файл с данными для анализа")

    file = get_file()
    if file:
        data = pd.read_csv(file, sep=',')

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
            #data = new_data
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

                    if (selected_y):
                        st.success("Столбец {} выбран успешно".format(selected_y))

                        train, test = train_test_split(data, test_size=test_size_slider/100)
                        #st.write(test_size_slider/100)
                        trainX = np.array(train.drop(selected_y, 1))
                        trainY = np.array(train[selected_y])
                        testX = np.array(test.drop(selected_y, 1))
                        testY = np.array(test[selected_y])

                        st.write(selected_y)
                       # st.write(trainY.reshape(trainY.shape[0]))

                        if reg_type == "Все":
                            reg_type = "Линейная регрессия"
                            n_predictions = model_linear_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                            predictions = model_linear_Regression(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                            reg_type = "Ridge"
                            n_predictions = model_ridge_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                            predictions = model_ridge_Regression(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                            reg_type = "Случайный лес"
                            n_predictions = model_random_forest(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                            predictions = model_random_forest(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                            reg_type = "Lasso"
                            n_predictions = model_lasso(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                            predictions = model_lasso(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                            reg_type=''


                        if reg_type == "Линейная регрессия":
                            n_predictions = model_linear_Regression(preprocessing.normalize(trainX), trainY, preprocessing.normalize(testX), testY)
                            predictions = model_linear_Regression(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)


                        if reg_type == "Ridge":
                            n_predictions = model_ridge_Regression(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX), testY)
                            predictions = model_ridge_Regression(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                        if reg_type == "Lasso":

                            n_predictions = model_lasso(preprocessing.normalize(trainX), trainY,preprocessing.normalize(testX),testY)
                            predictions = model_lasso(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)

                        if reg_type == "Случайный лес":
                            n_predictions = model_random_forest(preprocessing.normalize(trainX), trainY, preprocessing.normalize(testX), testY)
                            predictions = model_random_forest(trainX, trainY, testX, testY)
                            regression_plot_show(trainX, testY, predictions, n_predictions, reg_type, selected_y)
                            regression_errors_values(testY, predictions, n_predictions)

                            save_train_test_data_xlsx(trainX, trainY, testX, testY, predictions)
                    else:
                        st.warning("Выберите стоббец у")



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
                                cmap=plt.cm.get_cmap('nipy_spectral', 10))
                    plt.colorbar()
                    plt.title('Feautures PCA projection')
                    st.pyplot(plt.show())

                if st.checkbox("TSNE"):
                    tsne = TSNE(random_state=17)
                    tsne_representation = tsne.fit_transform(X)
                    plt.figure(figsize=(6, 6))
                    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1],
                                edgecolor='none', alpha=0.7, s=40, c=dataset['cluster'],
                                cmap=plt.cm.get_cmap('nipy_spectral', 10))
                    plt.colorbar()
                    plt.title('Feautures T-sne projection ')
                    st.pyplot(plt.show())

                save_exel(dataset,"Сохранить результаты xlsx File")

            elif problem_type == "Классификация":
                classification_type = st.radio("Выберите алгоритм решения",
                                    ("Все","KNeighborsClassifier","SVC_model" ))
                st.write("Выберите столбец у для задачи классификации")
                selected_y = st.multiselect("", all_clummn_names)
                if st.button("Выбрать"):
                    st.success("Столбец {} выбран успешно".format(selected_y))

                # ".iloc" принимает row_indexer, column_indexer
                y = np.array(data[selected_y])
                X = np.array(data.drop(selected_y, 1))

                # test_size показывает, какой объем данных нужно выделить для тестового набора
                # Random_state — просто сид для случайной генерации
                # Этот параметр можно использовать для воссоздания определённого результата:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

                if classification_type == "Все":
                    SVC_model = svm.SVC()
                    st.write("В KNN-модели нужно указать параметр n_neighbors. Это число точек, на которое будет смотреть классификатор, чтобы определить, к какому классу принадлежит новая точка")                 #
                    nbr = st.slider("Число точек ",3,data.shape[0])
                    KNN_model = KNeighborsClassifier(n_neighbors=nbr)
                    SVC_model.fit(X_train, y_train)
                    KNN_model.fit(X_train, y_train)

                    SVC_prediction = SVC_model.predict(X_test)
                    KNN_prediction = KNN_model.predict(X_test)

                    # Оценка точности — простейший вариант оценки работы классификатора
                    st.write("Оценка точности классификатора SVC_model")
                    st.write(accuracy_score(SVC_prediction, y_test))
                    st.write("Матрица неточности и отчёт о классификации дадут больше информации о производительности")
                    st.write(confusion_matrix(SVC_prediction, y_test))

                    st.write(classification_report(SVC_prediction, y_test))
                    st.write(SVC_prediction.tolist())

                    st.write("Оценка точности классификатора KNeighborsClassifier")
                    st.write(accuracy_score(KNN_prediction, y_test))
                    st.write("Матрица неточности и отчёт о классификации дадут больше информации о производительности")
                    # Но матрица неточности и отчёт о классификации дадут больше информации о производительности
                    st.write(confusion_matrix(KNN_prediction, y_test))
                    st.write(classification_report(KNN_prediction, y_test))
                    st.write(KNN_prediction.tolist())

                if classification_type == "KNeighborsClassifier":

                    st.write("В KNN-модели нужно указать параметр n_neighbors. Это число точек, на которое будет смотреть классификатор, чтобы определить, к какому классу принадлежит новая точка")                 #
                    nbr = st.slider("Число точек ",3,data.shape[0])
                    KNN_model = KNeighborsClassifier(n_neighbors=nbr)
                    KNN_model.fit(X_train, y_train)
                    KNN_prediction = KNN_model.predict(X_test)


                    st.write("Оценка точности классификатора KNeighborsClassifier")
                    st.write(accuracy_score(KNN_prediction, y_test))
                    st.write("Матрица неточности и отчёт о классификации дадут больше информации о производительности")
                    # Но матрица неточности и отчёт о классификации дадут больше информации о производительности
                    st.write(confusion_matrix(KNN_prediction, y_test))
                    st.write(classification_report(KNN_prediction, y_test))
                    st.write(KNN_prediction.tolist())

                if classification_type == "SVC_model":
                    SVC_model = svm.SVC()
                    SVC_model.fit(X_train, y_train)

                    SVC_prediction = SVC_model.predict(X_test)
                    # Оценка точности — простейший вариант оценки работы классификатора
                    st.write("Оценка точности классификатора SVC_model")
                    st.write(accuracy_score(SVC_prediction, y_test))
                    st.write("Матрица неточности и отчёт о классификации дадут больше информации о производительности")
                    st.write(confusion_matrix(SVC_prediction, y_test))
                    st.write(classification_report(SVC_prediction, y_test))
                    st.write(SVC_prediction.tolist())


        if st.button("Завершить работу"):
            st.balloons()
if __name__=='__main__':
    main()


