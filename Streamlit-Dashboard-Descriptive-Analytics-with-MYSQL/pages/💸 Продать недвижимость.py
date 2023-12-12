import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
import io
from sqlalchemy import create_engine
from datetime import datetime

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Sale", page_icon="💵", layout="wide")

st.title("Продажа недвижимости")

st.subheader(
    "Чтобы выставить на продажу квартиру - заполните необходимый раздел ниже.")

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.sidebar.image("data/logo1.png", caption="Work done by Alekseeva Ekaterina INBO-06-20")

db_connection_str = 'mysql+pymysql://root:2sm77XL0.@127.0.0.1:3306/streamlitkr'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM sberdata', con=db_connection)
df = pd.DataFrame(df, columns=['Unnamed', 'timestamp', 'price_doc', 'full_sq', 'life_sq', 'floor',
                               'max_floor', 'build_year', 'num_room', 'product_type', 'sub_area',
                               'raion_popul', 'green_zone_part', 'metro_min_avto', 'metro_km_avto',
                               'kremlin_km'])

col11, col12 = st.columns([1, 1])
with col11:
    st.subheader("Продажа для физического лица.")
    st.write("Заполните данные по Вашей квартире ниже.")

    ###############################
    ########## TIMESTAMP ##########
    ###############################

    timestamp = st.date_input("Выберите дату, когда опубликовать Вашу недвижимость на продажу. Если не требуется отложенное предложение - не заполняйте данное поле.", value=None)
    if timestamp == None:
        timestamp = datetime.now().date()
    st.write("Выбранная дата публикации: ", timestamp)

    ###############################
    ########## PRICE ##############
    ###############################

    price_doc = st.number_input('Введите стоимость квартиры')
    st.write('Выбранная цена квартиры:', price_doc)

    ###############################
    ########## SQUARES ############
    ###############################

    default_full_sq = [45.0]
    full_sq = st.slider("Укажите метраж квартиры, включая балконы, кухню и т.д.", min_value=1.0,
                                max_value=150.0,
                                value=default_full_sq, step=1.0)

    default_life_sq = [27.0]
    life_sq = st.slider("Укажите метраж квартиры, только жилую площадь:", min_value=1.0, max_value=150.0,
                                value=default_life_sq, step=1.0)

    ###############################
    ########### FLOOR #############
    ###############################

    default_floor = [6.0]
    def get_unique_floor(df):
        return np.unique(df.floor).tolist()


    unique_floor = get_unique_floor(df)
    min_floor = min(unique_floor)
    max_floor = max(unique_floor)

    floor = st.slider("Укажите этаж, на котором находится квартира:", min_value=min_floor, max_value=max_floor,
                              value=default_floor, step=1.0)

    ###############################
    ########### MAX_FLOOR #########
    ###############################

    default_max_floor = [9.0]
    def get_unique_max_floor(df):
        return np.unique(df.floor).tolist()


    unique_maxfloor = get_unique_max_floor(df)
    min_maxfloor = min(unique_maxfloor)
    max_maxfloor = max(unique_maxfloor)

    max_floor_switcher = st.slider("Выберите количество этажей в доме", min_value=min_maxfloor,
                                           max_value=max_maxfloor,
                                           value=default_max_floor, step=1.0)

    ###############################
    ########### YEAR ##############
    ###############################

    default_build_year = [1970.0]
    build_year = st.slider('Выберите год постройки дома:', min_value=1800.0, max_value=2020.0,
                                   value=default_build_year, step=1.0)

    ###############################
    ########### NUM_ROOMS #########
    ###############################

    default_num_rooms = [2.0]
    num_room = st.slider('Укажите количество комнат:', min_value=1.0, max_value=15.0,
                                 value=default_num_rooms, step=1.0)

    ###############################
    ########### PRODUCT_TYPE ######
    ###############################

    product_type = 'Investment'

    ###############################
    ########### DISTRICT ##########
    ###############################

    default_district = ['Alekseevskoe']
    district = st.multiselect(
        "Выберите ваш район:",
        options=df["sub_area"].sort_values().unique(),
        default=default_district
    )

    ###############################
    ########### RAION_POPUL #######
    ###############################

    raion_popul = st.number_input('Введите примерное заселенность Вашего района:')
    st.write('Заселеность вашего района:', raion_popul)

    ###############################
    #####GREEN_ZONE_PART ##########
    ###############################

    green_zone_part = st.number_input('Введите примерное доля площади зелени в общей площади Вашего района:')
    st.write('Заселеность вашего района:', green_zone_part)

    ###############################
    ###### METRO_MIN and KM #######
    ###############################

    metro_min_avto = st.number_input('Введите примерное время в минутах до ближайшего метро на машине:')
    st.write('Заселеность вашего района:', metro_min_avto)

    metro_km_avto = st.number_input('Введите примерное расстояние в километрах до ближайшего метро на машине:')
    st.write('Заселеность вашего района:', metro_km_avto)

    ###############################
    ###### KREMLIN DIST ###########
    ###############################

    kremlin_km = st.number_input('Введите примерное расстояние в километрах до центра города:')
    st.write('Ваша квартира располагается на таком расстоянии от центра:', kremlin_km)

    dict_df = {
        'timestamp': timestamp,
        'price_doc': price_doc,
        'full_sq': full_sq,
        'life_sq': life_sq,
        'floor': floor,
        'max_floor': max_floor_switcher,
        'build_year': build_year,
        'num_room': num_room,
        'product_type': product_type,
        'sub_area': district,
        'raion_popul': raion_popul,
        'green_zone_part': green_zone_part,
        'metro_min_avto': metro_min_avto,
        'metro_km_avto': metro_km_avto,
        'kremlin_km': kremlin_km
    }

    df_db = pd.DataFrame.from_dict(dict_df)

    if st.button('Добавить объявление о продаже', type='primary', key='button1'):
        st.write(df_db)
        try:
            df_db.to_sql('sberdata', con=db_connection, if_exists='append', index=False)
            st.success('Заявление успешно добавленно в базу данных недвижимости.')
        except Exception as e:
            st.error(f"Ошибка при добавлении заявки в базу данных: {e}")


with col12:
    st.subheader("Продажа для юридических лиц.")
    st.write("Образец для заполнения можно скачать ниже.")

    @st.cache_data
    def convert_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    example = pd.DataFrame(columns=['timestamp', 'price_doc', 'full_sq', 'life_sq', 'floor', 'max_floor', 'build_year',
                                    'num_room', 'product_type', 'sub_area', 'raion_popul', 'green_zone_part',
                                    'metro_min_avto', 'metro_km_avto', 'kremlin_km'])


    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        example.to_excel(writer, sheet_name='Книга 1', index=False)
        writer.close()

        download_example = st.download_button(
            label="Скачать образец",
            data=buffer,
            file_name='Example.xlsx',
            mime='application/vnd.ms-excel'
        )

    st.write("Загрузите заполненный образец с данными квартир клиентов в формате csv. ")

    uploaded_file = st.file_uploader("Выберите файл в формате csv.")
    if uploaded_file is not None:
        # upload_df = pd.read_csv(uploaded_file)
        upload_excel = pd.read_excel(uploaded_file)
        upload_df = pd.DataFrame(upload_excel)
        st.dataframe(upload_df.head(5))

    if st.button('Добавить объявление о продаже.', type='primary', key='button2'):
        try:
            upload_df.to_sql('sberdata', con=db_connection, if_exists='append', index=False)
            st.success('Заявление успешно добавленно в базу данных недвижимости.')
        except Exception as e:
            st.error(f"Ошибка при добавлении заявки в базу данных: {e}")

