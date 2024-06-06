import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.title('Tree Classifier')

uploaded_file = st.file_uploader("Choose a file")

# For test
# uploaded_file = './example_csv/wine_quality_data.csv'

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.title('Data Exploration')

    with st.expander("DataFrame"):
        st.dataframe(df)

    with st.expander("Summary Statistics"):
        st.write(df.describe())

    with st.expander("Missing Values"):
        st.write(df.isnull().sum())

    with st.expander("Correlation Matrix"):
        st.write(df.corr())

    st.divider()
    st.title('Data Preparation')

    target = st.selectbox(
        'Select the target',
        df.columns.tolist()
    )

    if target:
        available_features = [col for col in df.columns if col != target]
        features = st.multiselect(
            'Select the features',
            available_features,
            available_features
        )
    else:
        features = []
    
# Data preprocessing section
    st.divider()
    st.title('Data Preprocessing')

    missing_value_option = st.selectbox(
        'Select how to handle missing values',
        ['Drop rows with missing values', 'Fill missing values with mean', 'Fill missing values with median', 'Fill missing values with mode']
    )

    scaling_option = st.selectbox(
        'Select scaling method for numerical features',
        ['None', 'StandardScaler', 'MinMaxScaler']
    )

    encoding_option = st.selectbox(
        'Select encoding method for categorical features',
        ['None', 'OneHotEncoder']
    )

    if st.button('Apply Preprocessing'):
        # Missing value handling
        with st.expander("Data after missing value handling"):
            if missing_value_option == 'Drop rows with missing values':
                df = df.dropna()
            elif missing_value_option == 'Fill missing values with mean':
                df = df.fillna(df.mean())
            elif missing_value_option == 'Fill missing values with median':
                df = df.fillna(df.median())
            elif missing_value_option == 'Fill missing values with mode':
                df = df.fillna(df.mode().iloc[0])

            st.dataframe(df)

        # Creating pipelines for scaling and encoding
        with st.expander("Data after scaling and encoding"):
            numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = df[features].select_dtypes(include=['object']).columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler() if scaling_option == 'StandardScaler' else MinMaxScaler() if scaling_option == 'MinMaxScaler' else 'passthrough')
            ])

            categorical_transformer = Pipeline(steps=[
                ('encoder', OneHotEncoder() if encoding_option == 'OneHotEncoder' else 'passthrough')
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            df[features] = preprocessor.fit_transform(df[features])

            st.dataframe(df)

# Model training section
    test_size = st.slider('Test size', 0.1, 0.9, 0.2)

    df_train, df_test = train_test_split(df, test_size=test_size, random_state=1)

    st.divider()
    st.title('Modeling')

    model_option = st.selectbox(
        'Select the model',
        ['Decision Tree', 'Random Forest', 'Gradient Boosting']
    )

    if model_option == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=4)
    
    elif model_option == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=1)
    
    elif model_option == 'Gradient Boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=1)
    
    # If target is not selected, disable button
    if not target or not features:
        st.warning("Please select target and features")
        button = st.button('Train the model', disabled=True)
    else:
        button = st.button('Train the model')
    
    if button:
        model.fit(df_train[features], df_train[target])
        st.write('Model trained')

        st.divider()
        st.title('Evaluation')

        pred_train = model.predict(df_train[features])
        pred_test = model.predict(df_test[features])

        st.write('F1 Score')
        from sklearn.metrics import f1_score
        st.write('Train:', f1_score(df_train[target], pred_train))
        st.write('Test:', f1_score(df_test[target], pred_test))

# Model Visualization section
        # Confusion Matrix
        with st.expander("Confusion Matrix"):
            cm = confusion_matrix(df_test[target], pred_test)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        # Classification Report
        with st.expander("Classification Report"):
            report = classification_report(df_test[target], pred_test, output_dict=True)
            st.write(report)

        # ROC Curve and AUC
        if len(df[target].unique()) == 2:
            # If the target has only 2 classes
            with st.expander("ROC Curve"):
                fpr, tpr, _ = roc_curve(df_test[target], model.predict_proba(df_test[features])[:, 1])
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc='lower right')
                st.pyplot(fig)

        # Feature Importances
        if model_option in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
            with st.expander("Feature Importances"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                fig, ax = plt.subplots()
                ax.bar(range(len(features)), importances[indices])
                ax.set_xticks(range(len(features)))
                ax.set_xticklabels([features[i] for i in indices], rotation=90)
                st.pyplot(fig)

        # Learning Curve
        with st.expander("Learning Curve"):
            train_sizes, train_scores, test_scores = learning_curve(model, df[features], df[target], cv=5)
            train_scores_mean = np.mean(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)

            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_scores_mean, label='Train score')
            ax.plot(train_sizes, test_scores_mean, label='Test score')
            ax.set_xlabel('Training examples')
            ax.set_ylabel('Score')
            ax.legend(loc='best')
            st.pyplot(fig)
