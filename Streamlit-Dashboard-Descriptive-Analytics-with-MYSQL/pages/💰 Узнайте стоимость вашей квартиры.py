import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optuna.integration import shap
import shap
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from io import StringIO
from catboost import Pool
from PIL import Image
from sqlalchemy import create_engine
from streamlit_extras.metric_cards import style_metric_cards

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="CatBoost", page_icon="📈", layout="wide")

st.title("Рыночная стоимость вашей квартиры.")

st.subheader(
    "Получим предсказанную цену с помощью модели градиентного бустинга. Ниже приведены метрики качеств а используемой модели.")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.image("data/logo1.png", caption="Work done by Alekseeva Ekaterina INBO-06-20")
st.sidebar.title("Введите параметры своей квартиры")

# db
db_connection_str = 'mysql+pymysql://root:2sm77XL0.@127.0.0.1:3306/streamlitkr'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM sberdata', con=db_connection)
df = pd.DataFrame(df, columns=['Unnamed', 'timestamp', 'price_doc', 'full_sq', 'life_sq', 'floor',
                               'max_floor', 'build_year', 'num_room', 'product_type', 'sub_area',
                               'raion_popul', 'green_zone_part', 'metro_min_avto', 'metro_km_avto',
                               'kremlin_km'])

df = df[df['max_floor'] != 0]
selected_columns = [
    'full_sq', 'life_sq', 'floor', 'max_floor',
    'build_year', 'num_room', 'raion_popul',
    'green_zone_part', 'metro_min_avto', 'metro_km_avto', 'kremlin_km']
df_selected = df[selected_columns].dropna()

def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1

    df_clean = df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df_clean

df_cleaned = remove_outliers(df, selected_columns)

df_cleaned[['build_year', 'floor', 'max_floor']] = df_cleaned[['build_year', 'floor', 'max_floor']].astype('int64')

X = df_cleaned[['full_sq', 'life_sq', 'floor', 'max_floor',
    'build_year', 'num_room', 'raion_popul',
    'green_zone_part', 'metro_min_avto', 'metro_km_avto', 'kremlin_km']]
y = df_cleaned.price_doc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)


model = CatBoostRegressor()
model.load_model('CatBoostModel.cbm')

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

metric1, metric2, metric3, metric4, metric5 = st.columns(5, gap='small')
with metric1:
    st.info('MSE', icon="📌")
    st.metric(label="MSE", value=round(mse, 3))
with metric2:
    st.info('MAE', icon="📌")
    st.metric(label="MAE", value=round(mae, 3))
with metric3:
    st.info('R2', icon="📌")
    st.metric(label="R2", value=round(r2, 3))
with metric4:
    st.info('RMSE', icon="📌")
    st.metric(label="RMSE", value=round(rmse, 3))
with metric5:
    st.info('MAPE', icon="📌")
    st.metric(label="MAPE", value=round(mape, 3))

style_metric_cards(background_color="#172d43", border_left_color="#ff4b4b",
                   border_color="#1f66bd", box_shadow="#F71938")

def shap_beeswarm():
    test_pool = Pool(X_test, label=y_test, cat_features=['build_year', 'floor', 'max_floor'])

    explainer = shap.Explainer(model=model, seed=555, feature_names=test_pool.get_feature_names())
    shap_values = explainer(X_test, y=test_pool.get_label())
    shap.initjs()

    plt.rcParams['savefig.transparent']='True'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    shap.plots.beeswarm(shap_values=shap_values)

    plt.savefig('shap_beeswarm_plot.png', bbox_inches = 'tight')
    plt.close()

shap_beeswarm()
image = Image.open('shap_beeswarm_plot.png')

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Ниже вы видите какие параметры и как влияют на цены квартир.")
    st.image(image, caption='SHAP Beeswarm Plot', width=1000)
with col2:
    st.subheader("Интерпретация графика")
    st.write("Как мы можем видеть из графика, модель выбирает наиболее подходящие признаки для прогнозирования цены недвижимости.")
    st.write(
        "Чем больше дисперсия по признакам - тем больше влияния оказывает рассматриваемый признак на таргет значение "
        "модели. Цвета точек влияют на 'направление' предсказания, то есть синие точки - уменьшают таргет, "
        "а красные - увеличивают, в нашем случае, цену квартиры. Значения на оси абцисс нормализованные. Показывают непосредственно значения признаков.")

    st.write(
        "Из графика видим, что наиболее значимый вклад в прогноз вносят признаки: full_sq (общая площадь квартиры), kremlin_km (расстояние до центра города), build_year (год постройки) и life_sq (жилая площадь).")

    st.write(
        "Рассмотрим первый признак, метраж квартиры (признак full_sq). Мы видим, что чем меньше общая площадь квартиры, тем меньше на нее цена (голубые точки слева от ординаты в 0).  Аналогично, чем больше значение цены - тем в большую сторону изменяется таргет значение, то есть цена.")

    st.write(
        "Аналогично со второй строкой, показывающей расстояние до центра города, чем меньше расстояние до центра города (красные точки слева от ординаты в 0) - тем больше цена и наоборот.")

    st.write(
        "В заключении, рассмотрим признак build_year, год постройки дома. Мы видим, что красные точки есть и справа и слева от оси ординат в 0. Это значит, что очень старые дома и новые стоят дороже, чем дома построенные в середине диапазона, в 1960-90х годах.")

default_district = ['Severnoe Butovo']
district = st.sidebar.multiselect(
    "Выберите ваш район:",
    options=df["sub_area"].sort_values().unique(),
    default=default_district
)

default_max_floor = [9.0]
def get_unique_max_floor(df):
    return np.unique(df.floor).tolist()


unique_maxfloor = get_unique_max_floor(df)
min_maxfloor = min(unique_maxfloor)
max_maxfloor = max(unique_maxfloor)

max_floor_switcher = st.sidebar.slider("Выберите количество этажей в доме", min_value=min_maxfloor,
                                       max_value=max_maxfloor,
                                       value=default_max_floor, step=1.0)


default_floor = [6.0]
def get_unique_floor(df):
    return np.unique(df.floor).tolist()


unique_floor = get_unique_floor(df)
min_floor = min(unique_floor)
max_floor = max(unique_floor)

floor = st.sidebar.slider("Выберите интересующий этаж:", min_value=min_floor, max_value=max_floor,
                          value=default_floor, step=1.0)

default_full_sq = [45.0]
full_sq = st.sidebar.slider("Укажите метраж квартиры, включая балконы, кухню и т.д.", min_value=1.0, max_value=150.0,
                            value=default_full_sq, step=1.0)

default_life_sq = [27.0]
life_sq = st.sidebar.slider("Выберите метраж квартиры, только жилую площадь:", min_value=1.0, max_value=150.0,
                            value=default_life_sq, step=1.0)

default_num_rooms = [2.0]
num_room = st.sidebar.slider('Выберите количество комнат:', min_value=1.0, max_value=15.0,
                             value=default_num_rooms, step=1.0)

default_build_year = [1970.0]
build_year = st.sidebar.slider('Выберите год постройки дома:', min_value=1800.0, max_value=2020.0,
                             value=default_build_year, step=1.0)

raion_popul = st.sidebar.number_input('Введите примерное заселенность Вашего района:')
st.sidebar.write('Заселеность вашего района:', raion_popul)

green_zone_part = st.sidebar.number_input('Введите примерное доля площади зелени в общей площади Вашего района:')
st.sidebar.write('Заселеность вашего района:', green_zone_part)

metro_min_avto = st.sidebar.number_input('Введите примерное время в минутах до ближайшего метро на машине:')
st.sidebar.write('Заселеность вашего района:', metro_min_avto)

metro_km_avto = st.sidebar.number_input('Введите примерное расстояние в километрах до ближайшего метро на машине:')
st.sidebar.write('Заселеность вашего района:', metro_km_avto)

kremlin_km = st.sidebar.number_input('Введите примерное расстояние в километрах до центра города:')
st.sidebar.write('Ваша квартира располагается на таком расстоянии от центра:', kremlin_km)

dict_data = {
    'full_sq': full_sq,
    'life_sq': life_sq,
    'floor': floor,
    'max_floor': max_floor_switcher,
    'build_year': build_year,
    'num_room': num_room,
    'raion_popul': raion_popul,
    'green_zone_part': green_zone_part,
    'metro_min_avto': metro_min_avto,
    'metro_km_avto': metro_km_avto,
    'kremlin_km': kremlin_km
}

df_reg = pd.DataFrame.from_dict(dict_data)

st.subheader(
    "Введите в левое меню данные по Вашей квартире. Ниже подсчитается примерная рыночная стоимость.")


y1 = model.predict(df_reg)
our_price = y1.astype(int)[0]

txt = "👛 " + str(our_price) + " ₽"
htmlstr1 = f"""<p style='background-color:#172d43;
                                           color:white;
                                           font-size:54px;
                                           border-radius:15px;
                                           line-height:60px;
                                           padding:17px;
                                           opacity:1.0;
                                           text-align: center'>
                                           {txt}</style>
                                           <br></p>"""
st.markdown(htmlstr1, unsafe_allow_html=True)

def graphs():
    test_pool = Pool(data=df_reg, label=[our_price])

    explainer_new = shap.Explainer(model=model, seed=555, feature_names=test_pool.get_feature_names())
    shap_values_new = explainer_new(df_reg, y=test_pool.get_label())
    shap.initjs()

    shap.plots.force(shap_values_new[0], show=True)

    shap.save_html('shap_plot_new.html', shap.plots.force(shap_values_new[0]))

    with open('shap_plot_new.html', 'r', encoding='utf-8') as file:
        html_code_new = file.read()

    st.components.v1.html(html_code_new, height=300)

st.subheader(
    "Ниже вы видите какие параметры и как влияют на цену Вашей квартиры.")

graphs()


