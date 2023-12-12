import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from sqlalchemy import create_engine

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
st.title("Общая аналитика рынка")

theme_plotly = None  # None or streamlit

with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# db
db_connection_str = 'mysql+pymysql://root:2sm77XL0.@127.0.0.1:3306/streamlitkr'
db_connection = create_engine(db_connection_str)

df = pd.read_sql('SELECT * FROM sberdata', con=db_connection)
df = pd.DataFrame(df, columns=['Unnamed', 'timestamp', 'price_doc', 'full_sq', 'life_sq', 'floor',
                               'max_floor', 'build_year', 'num_room', 'product_type', 'sub_area',
                               'raion_popul', 'green_zone_part', 'metro_min_avto', 'metro_km_avto',
                               'kremlin_km'])
df = df[df['max_floor'] != 0]

# sidebar
st.sidebar.image("data/logo1.png", caption="Work done by Alekseeva Ekaterina INBO-06-20")
district = st.sidebar.multiselect(
    "Выберите интересующий район",
    options=df["sub_area"].sort_values().unique(),
    default=None,
)

max_floor_switcher = st.sidebar.multiselect(
    "Выберите количество этажей в доме",
    options=df["max_floor"].sort_values().unique(),
    default=None,
)
num_room = st.sidebar.multiselect(
    'Выберите количество комнат',
    options=df['num_room'].sort_values().unique(),
    default=None,
)


def get_unique_floor(df):
    return np.unique(df.floor).tolist()


unique_floor = get_unique_floor(df)
min_floor = min(unique_floor)
max_floor = max(unique_floor)

floor = st.sidebar.slider("Выберите интересующий этаж", min_value=min_floor, max_value=max_floor,
                          value=[min(unique_floor), max(unique_floor)])

st.sidebar.write('Вы выбрали:', floor)


def get_unique_build_year(df):
    return np.unique(df.build_year).tolist()


unique_build_year = get_unique_build_year(df)
min_build_year = min(unique_build_year)
max_build_year = max(unique_build_year)

build_year = st.sidebar.slider('Выберите год постройки дома', min_value=min_build_year, max_value=max_build_year,
                               value=[min(unique_build_year), max(unique_build_year)])

st.sidebar.write('Вы выбрали:', build_year)

df_selection = df.query(
    "sub_area == @district & max_floor == @max_floor_switcher &  @min_build_year <= build_year <= @max_build_year &  "
    "@min_floor <= floor <= @max_floor & num_room == @num_room")


def Home():
    with st.expander("Посмотреть подходящие предложения"):
        showData = st.multiselect('Filter: ', df_selection.columns,
                                  default=['timestamp', 'price_doc', 'full_sq', 'life_sq', 'floor', 'max_floor',
                                           'build_year', 'num_room', 'product_type', 'sub_area', 'raion_popul',
                                           'green_zone_part', 'metro_min_avto', 'metro_km_avto'])
        st.dataframe(df_selection[showData], use_container_width=True)

    # basic analysis
    total_offers = float(df_selection['product_type'].count())

    if not df_selection.empty:
        average_prices = df_selection.groupby('sub_area')['price_doc'].mean()
        expensive_district = average_prices.idxmax()
        cheap_district = average_prices.idxmin()
    else:
        expensive_district = 'Выберите условия'
        cheap_district = 'Выберите условия'

    if not df_selection.empty:
        price_mean = float(df_selection['price_doc'].mean())
        price_mean = round(price_mean, 2)
    else:
        price_mean = 'Выберите условия'

    total1, total2, total3, total4 = st.columns(4, gap='small')
    with total1:
        st.info('Всего предложений по запросу', icon="📌")
        st.metric(label="Total offes ", value=f"{total_offers:,.0f}")

    with total2:
        st.info('Район с самыми дорогими квартирами', icon="📌")
        st.metric(label="Most expensive district", value=expensive_district)

    with total3:
        st.info('Район с самыми дешевыми квартирами', icon="📌")
        st.metric(label="Cheapest district", value=cheap_district)

    with total4:
        st.info('Средняя цена', icon="📌")
        st.metric(label="Mean price", value=price_mean)

    style_metric_cards(background_color="#172d43", border_left_color="#ff4b4b", border_color="#1f66bd",
                       box_shadow="#F71938")

    # variable distribution Histogram
    with st.expander("🔻Гистограммы частот🔻"):
        df.hist(figsize=(16, 8), color='#00588E', zorder=2, rwidth=0.9, legend=['product_type']);
        st.pyplot()


# graphs
def graphs():
    # bar
    count_by_district = (
        df_selection['sub_area'].value_counts().reset_index()
    )

    fig_apartments_by_area = px.bar(count_by_district, x='count', y='sub_area', orientation='h',
                                    title='Количество доступных квартир в выбранных районах',
                                    # color_discrete_sequence = px.colors.sequential.Blues_r,
                                    labels={'count': 'Количество квартир в районе', 'sub_area': 'Район'},
                                    template='plotly_white',
                                    color_discrete_sequence=['#636efa'])

    fig_apartments_by_area.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=(dict(showgrid=False))
    )

    # line
    price_by_district = (
        df_selection.groupby('sub_area')['price_doc'].mean().reset_index()
    )
    fig_price_by_district = px.line(price_by_district, x='sub_area', y='price_doc', orientation='v',
                                    title='Средние цены в выбранных районах',
                                    # color='price_doc',  # цветовая схема по цене
                                    labels={'price_doc': 'Цена квартиры', 'sub_area': 'Район'},
                                    template='plotly_white',
                                    color_discrete_sequence=['#636efa'])

    fig_price_by_district.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False))
    )

    left, right, center = st.columns(3)
    left.plotly_chart(fig_price_by_district, use_container_width=True)
    right.plotly_chart(fig_apartments_by_area, use_container_width=True)

    with center:
        # pie chart
        fig = px.pie(df_selection, values='raion_popul', names='sub_area', title='Самые густо населенные районы')
        fig.update_layout(legend_title="Район", legend_y=0.9)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        fig.update_traces(marker=dict(colors=px.colors.sequential.Blues_r))
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)


def sideBar():
    Home()
    graphs()


sideBar()

# theme
hide_st_style = """  
 
<style> 
#MainMenu {visibility:hidden;} 
footer {visibility:hidden;} 
header {visibility:hidden;} 
</style> 
"""
