from datetime import date
import numpy as np
import pandas as pd
import pickle
import psycopg2
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings
warnings.filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Forecast', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Retail Sales Forecast</h1>',
                unsafe_allow_html=True)
    add_vertical_space(1)


# custom style for submit button - color and width

def style_submit_button():

    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)


# custom style for prediction result text - color and position

def style_prediction():

    st.markdown(
        """
            <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
        unsafe_allow_html=True
    )


# SQL columns ditionary

def columns_dict():

    columns_dict = {'day': 'Day', 'month': 'Month', 'year': 'Year', 'store': 'Store',
                    'dept': 'Dept', 'type': 'Type', 'weekly_sales': 'Weekly_Sales',
                    'size': 'Size', 'is_holiday': 'IsHoliday', 'temperature': 'Temperature',
                    'fuel_price': 'Fuel_Price', 'markdown1': 'MarkDown1', 'markdown2': 'MarkDown2',
                    'markdown3': 'MarkDown3', 'markdown4': 'MarkDown4', 'markdown5': 'MarkDown5',
                    'cpi': 'CPI', 'unemployment': 'Unemployment'}
    return columns_dict



class plotly:

    def vertical_bar_chart(df, x, y, text, color, title, title_x=0.25):

        fig = px.bar(df, x=x, y=y, labels={x: '', y: ''}, title=title)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(title_x=title_x, title_font_size=22)

        df[y] = df[y].astype(float)
        text_position = ['inside' if val >= max(
            df[y]) * 0.90 else 'outside' for val in df[y]]

        fig.update_traces(marker_color=color,
                          text=df[text],
                          textposition=text_position,
                          texttemplate='%{y}',
                          textfont=dict(size=14),
                          insidetextfont=dict(color='white'),
                          textangle=0,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)



class sql:

    def create_table():

        try:

            krish = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='5555',
                                    database='retail_forecast')
            cursor = krish.cursor()

            cursor.execute(f'''create table if not exists sales(
                                    day           	int,
                                    month           int,
                                    year            int,
                                    store           int,
                                    dept            int,
                                    type            int,
                                    weekly_sales    float,
                                    size            int,
                                    is_holiday      int,
                                    temperature     float,
                                    fuel_price      float,
                                    markdown1       float,
                                    markdown2       float,
                                    markdown3       float,
                                    markdown4       float,
                                    markdown5       float,
                                    cpi             float,
                                    unemployment    float);''')

            krish.commit()
            cursor.close()
            krish.close()
        
        except:
            
            st.warning("There is no database named 'retail_forecast'. Please create the database.")


    def drop_table():

        try:

            krish = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='5555',
                                    database='retail_forecast')
            cursor = krish.cursor()

            cursor.execute(f'''drop table if exists sales;''')

            krish.commit()
            cursor.close()
            krish.close()

        except:
            pass
    

    def data_migration():
        
        try:
            f = pd.read_csv(r'dataset\df_sql.csv')
            df = pd.DataFrame(f)

            krish = psycopg2.connect(host='localhost',
                                    user='postgres',
                                    password='5555',
                                    database='retail_forecast')
            cursor = krish.cursor()

            cursor.executemany(f'''insert into sales(day,month,year,store,dept,type,
                                                weekly_sales,size,is_holiday,
                                                temperature,fuel_price,markdown1,
                                                markdown2,markdown3,markdown4,
                                                markdown5,cpi,unemployment) 
                            values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            %s,%s,%s);''', df.values.tolist())
            krish.commit()
            cursor.close()
            krish.close()

        except Exception as e:
            st.warning(e)



class top_sales:

    # sales table order by date,store and dept

    def sql(condition):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select * from sales
                            where {condition}
                            order by year, month, day, store, dept asc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        krish.close()

        return df


    # year list from sales table 
    
    def year():
        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct year from sales
                           order by year asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # month list from sales table based on selected year

    def month(year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct month from sales
                           where year='{year}'
                           order by month asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # day list from sales table based on selected year, month

    def day(month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct day from sales
                           where year='{year}' and month='{month}'
                           order by day asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # store list from sales table based on selected year,month, day

    def store(day, month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct store from sales
                           where day='{day}' and year='{year}' and month='{month}'
                           order by store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # department list from sales table based on selected year,month, day, store
    
    def dept(day, month, year, store):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct dept from sales
                           where day='{day}' and month='{month}' and year='{year}' and store='{store}'
                           order by dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # department list (with 'Overall') from sales table based on selected year,month, day

    def top_store_dept(day, month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct dept from sales
                           where day='{day}' and month='{month}' and year='{year}' 
                           order by dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]
        data.insert(0, 'Overall')

        cursor.close()
        krish.close()

        return data


    # top 10 stores filter options

    def top_store_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            year = st.selectbox(label='Year ', options=top_sales.year())

        with col2:
            month = st.selectbox(label='Month ', options=top_sales.month(year))

        with col1:
            day = st.selectbox(
                label='Day ', options=top_sales.day(month, year))

        with col4:
            dept = st.selectbox(
                label='Dept ', options=top_sales.top_store_dept(day, month, year))

        return day, month, year, dept


    # top 10 stores based on weekly_sales

    def top_store_sales(condition):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select store, sum(weekly_sales) as weekly_sales
                            from sales
                            where {condition}
                            group by store
                            order by weekly_sales desc
                            limit 10;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['Store', 'Weekly Sales'], index=index)
        df['Weekly Sales'] = df['Weekly Sales'].apply(lambda x: f"{x:.2f}")
        df['store_x'] = df['Store'].apply(lambda x: str(x)+'*')
        df = df.rename_axis('s.no')

        cursor.close()
        krish.close()

        return df


    # store list (with 'Overall') from sales table

    def top_dept_store(day, month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct store from sales
                           where day='{day}' and month='{month}' and year='{year}' 
                           order by store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]
        data.insert(0, 'Overall')

        cursor.close()
        krish.close()

        return data


    # top 10 departments filter options

    def top_dept_filter_options():

        col1, col2, col3, col4 = st.columns(4, gap='medium')

        with col3:
            year = st.selectbox(label='Year  ', options=top_sales.year())

        with col2:
            month = st.selectbox(
                label='Month  ', options=top_sales.month(year))

        with col1:
            day = st.selectbox(
                label='Day  ', options=top_sales.day(month, year))

        with col4:
            store = st.selectbox(
                label='Store  ', options=top_sales.top_dept_store(day, month, year))

        return day, month, year, store


    # top 10 departments based on weekly_sales

    def top_dept_sales(condition):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select dept, sum(weekly_sales) as weekly_sales
                            from sales
                            where {condition}
                            group by dept
                            order by weekly_sales desc
                            limit 10;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        df = pd.DataFrame(s, columns=['Dept', 'Weekly Sales'], index=index)
        df['Weekly Sales'] = df['Weekly Sales'].apply(lambda x: f"{x:.2f}")
        df['dept_x'] = df['Dept'].apply(lambda x: str(x)+'*')
        df = df.rename_axis('s.no')

        cursor.close()
        krish.close()

        return df



class comparison:

    # sales table order by date,store,dept

    def sql(condition):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select * from sales
                            where {condition}
                            order by year, month, day, store, dept asc;''')

        s = cursor.fetchall()

        index = [i for i in range(1, len(s)+1)]
        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns, index=index)
        df = df.rename_axis('s.no')

        cursor.close()
        krish.close()

        return df


    # vertical line for st.metrics()

    def vertical_line():
        line_width = 2
        line_color = 'grey'

        # Use HTML and CSS to create the vertical line
        st.markdown(
            f"""
            <style>
                .vertical-line {{
                    border-left: {line_width}px solid {line_color};
                    height: 100vh;
                    position: absolute;
                    left: 55%;
                    margin-left: -{line_width / 2}px;
                }}
            </style>
            <div class="vertical-line"></div>
            """,
            unsafe_allow_html=True
        )


    # year list from sales table

    def year():
        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct year from sales
                           order by year asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # month list from sales table based on year

    def month(year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct month from sales
                           where year='{year}'
                           order by month asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # day list from sales table based on year,month

    def day(month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct day from sales
                           where year='{year}' and month='{month}'
                           order by day asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # store list from sales table based on year,month,day

    def store(day, month, year):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct store from sales
                           where day='{day}' and year='{year}' and month='{month}'
                           order by store asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # department list from sales table based on year,month,day,store

    def dept(day, month, year, store):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct dept from sales
                           where day='{day}' and month='{month}' and year='{year}' and store='{store}'
                           order by dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # manual tab filter options

    def manual_filter_options():

        col16, col17 = st.columns(2, gap='large')

        with col16:

            col6, col7, col8 = st.columns(3)

            with col8:
                year1 = st.selectbox(label='Year1', options=comparison.year())

            with col7:
                month1 = st.selectbox(
                    label='Month1', options=comparison.month(year1))

            with col6:
                day1 = st.selectbox(
                    label='Day1', options=comparison.day(month1, year1))

            col9A, col9, col10, col10A = st.columns([0.1, 0.4, 0.4, 0.1])

            with col9:
                store1 = st.selectbox(
                    label='Store1', options=comparison.store(day1, month1, year1))

            with col10:
                dept1 = st.selectbox(label='Dept1', options=comparison.dept(
                    day1, month1, year1, store1))

        with col17:

            col11, col12, col13 = st.columns(3)

            with col13:
                year2 = st.selectbox(label='Year2', options=comparison.year())

            with col12:
                month2 = st.selectbox(
                    label='Month2', options=comparison.month(year2))

            with col11:
                day2 = st.selectbox(
                    label='Day2', options=comparison.day(month2, year2))

            col14A, col14, col15, col15A = st.columns([0.1, 0.4, 0.4, 0.1])

            with col14:
                manual_store = comparison.store(day2, month2, year2)
                manual_store[0], manual_store[1] = manual_store[1], manual_store[0]
                store2 = st.selectbox(label='Store2', options=manual_store)

            with col15:
                if year1 == year2 and month1 == month2 and day1 == day2 and store1 == store2:
                    dept = comparison.dept(day2, month2, year2, store2)
                    dept.remove(dept1)
                    dept2 = st.selectbox(label='Dept2', options=dept)
                else:
                    dept2 = st.selectbox(label='Dept2', options=comparison.dept(
                        day2, month2, year2, store2))

        return day1, month1, year1, store1, dept1, day2, month2, year2, store2, dept2


    # comparison between 2 different stores and department combination

    def manual_comparison(df1, df2):

        data1 = df1.iloc[0, :]
        df1_dict = data1.to_dict()

        data2 = df2.iloc[0, :]
        df2_dict = data2.to_dict()

        col1, col2, col3 = st.columns([0.1, 0.9, 0.1])
        with col2:
            holiday = {0: 'No', 1: 'Yes'}
            type = {1: 'A', 2: 'B', 3: 'C'}
            st.code(f'''{type[df1_dict['type']]} : Type : {type[df2_dict['type']]}           {df1_dict['size']} : Size : {df2_dict['size']}           {holiday[df1_dict['is_holiday']]}  :  Holiday : {holiday[df2_dict['is_holiday']]}''')

        comparison.vertical_line()
        col1, col2 = st.columns([0.55, 0.45])

        with col1:

            for i in ['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']:
                p, c = df1_dict[i], df2_dict[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col3, col4, col5, col6 = st.columns([0.3, 0.4, 0.25, 0.05])
                    with col3:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col4:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col5:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")

        with col2:

            for i in ['markdown1', 'markdown2', 'markdown3', 'markdown4', 'markdown5']:
                p, c = df1_dict[i], df2_dict[i]
                diff = ((c-p)/p)*100

                if p != 0:
                    col7, col8, col9, col10 = st.columns(
                        [0.05, 0.3, 0.4, 0.25])
                    with col8:
                        add_vertical_space(1)
                        st.markdown(f"#### {df1_dict[i]:.2f}")

                    with col9:
                        if f"{c:.2f}" == f"{p:.2f}":
                            add_vertical_space(1)
                            st.markdown(f'''<h5 style="text-align: left;">{columns_dict()[i]} <br>
                                        No Impact</h5>''', unsafe_allow_html=True)
                            add_vertical_space(2)
                        else:
                            st.metric(
                                label=columns_dict()[i], value=f"{(c-p):.2f}", delta=f"{diff:.2f}%")

                    with col10:
                        add_vertical_space(1)
                        st.markdown(f"#### {df2_dict[i]:.2f}")


class prediction:

    # type and size dictionary based on store from sales table --> {store:type},{store:size}

    def type_size_dict():

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct store, type, size 
                           from sales
                           group by store, type, size
                           order by store asc;''')

        s = cursor.fetchall()

        columns = [i[0] for i in cursor.description]

        df = pd.DataFrame(s, columns=columns)

        store = df['store'].to_list()
        type = df['type'].to_list()
        size = df['size'].to_list()

        type_dict, size_dict = {}, {}

        for i in range(0, len(store)):
            type_dict[store[i]] = type[i]
            size_dict[store[i]] = size[i]

        cursor.close()
        krish.close()

        return type_dict, size_dict


    # department list from sales table based on store

    def dept(store):

        krish = psycopg2.connect(host='localhost',
                                user='postgres',
                                password='5555',
                                database='retail_forecast')
        cursor = krish.cursor()

        cursor.execute(f'''select distinct dept from sales
                           where store='{store}'
                           order by dept asc;''')

        s = cursor.fetchall()

        data = [i[0] for i in s]

        cursor.close()
        krish.close()

        return data


    # get input data from users and predict weekly_sales
    
    def predict_weekly_sales():

        # get input from users
        with st.form('prediction'):

            col1, col2, col3 = st.columns([0.5, 0.1, 0.5])

            with col1:

                user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                          max_value=date(2013, 12, 31), value=date(2010, 2, 5))

                store = st.number_input(label='Store', min_value=1, max_value=45,
                                        value=1, step=1)

                dept = st.selectbox(label='Department',
                                    options=prediction.dept(store))

                holiday = st.selectbox(label='Holiday', options=['Yes', 'No'])

                temperature = st.number_input(label='Temperature(°F)', min_value=-10.0,
                                              max_value=110.0, value=-7.29)

                fuel_price = st.number_input(label='Fuel Price', max_value=10.0,
                                             value=2.47)

                cpi = st.number_input(label='CPI', min_value=100.0,
                                      max_value=250.0, value=126.06)

            with col3:

                markdown1 = st.number_input(label='MarkDown1', value=-2781.45)

                markdown2 = st.number_input(label='MarkDown2', value=-265.76)

                markdown3 = st.number_input(label='MarkDown3', value=-179.26)

                markdown4 = st.number_input(label='MarkDown4', value=0.22)

                markdown5 = st.number_input(label='MarkDown5', value=-185.87)

                unemployment = st.number_input(label='Unemployment',
                                               max_value=20.0, value=3.68)

                add_vertical_space(2)
                button = st.form_submit_button(label='SUBMIT')
                style_submit_button()

        # user entered the all input values and click the button
        if button:
            with st.spinner(text='Processing...'):

                # load the regression pickle model
                with open(r'model1_markdown.pkl', 'rb') as f:
                    model = pickle.load(f)

                holiday_dict = {'Yes': 1, 'No': 0}
                type_dict, size_dict = prediction.type_size_dict()

                # make array for all user input values in required order for model prediction
                user_data = np.array([[user_date.day, user_date.month, user_date.year,
                                       store, dept, type_dict[store], size_dict[store],
                                       holiday_dict[holiday], temperature,
                                       fuel_price, markdown1, markdown2, markdown3,
                                       markdown4, markdown5, cpi, unemployment]])

                # model predict the selling price based on user input
                y_pred = model.predict(user_data)[0]

                # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
                weekly_sales = f"{y_pred:.2f}"

                return weekly_sales




streamlit_config()


with st.sidebar:

    add_vertical_space(1)
    option = option_menu(menu_title='', options=['Migrating to SQL', 'Sales Analysis', 'Comparison', 'Prediction', 'Exit'],
                         icons=['database-fill', 'bar-chart-line', 'globe', 'list-task', 'slash-square', 'sign-turn-right-fill'])
    
    col1, col2, col3 = st.columns([0.26, 0.48, 0.26])
    with col2:
        button = st.button(label='Submit')



if button and option == 'Migrating to SQL':

    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

    with col2:

        add_vertical_space(2)

        with st.spinner('Dropping the Existing Table...'):
            sql.drop_table()
        
        with st.spinner('Creating Sales Table...'):
            sql.create_table()
        
        with st.spinner('Migrating Data to SQL Database...'):
            sql.data_migration()

        st.success('Successfully Data Migrated to SQL Database')
        st.balloons()



elif option == 'Sales Analysis':

    tab1, tab2 = st.tabs(['Stores', 'Departments'])

    with tab1:

        day1, month1, year1, dept1 = top_sales.top_store_filter_options()
        add_vertical_space(3)

        if dept1 == 'Overall':
            df1 = top_sales.top_store_sales(
                f"day='{day1}' and month='{month1}' and year='{year1}'")

            plotly.vertical_bar_chart(df=df1, x='store_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#5D9A96', title='Top Stores in Weekly Sales', title_x=0.35)

        else:
            df1 = top_sales.top_store_sales(
                f"day='{day1}' and month='{month1}' and year='{year1}' and dept='{dept1}'")

            plotly.vertical_bar_chart(df=df1, x='store_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#5D9A96', title='Top Stores in Weekly Sales', title_x=0.35)


    with tab2:

        day2, month2, year2, store2 = top_sales.top_dept_filter_options()
        add_vertical_space(3)

        if store2 == 'Overall':
            df2 = top_sales.top_dept_sales(
                f"day='{day2}' and month='{month2}' and year='{year2}'")

            plotly.vertical_bar_chart(df=df2, x='dept_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#5cb85c', title='Top Departments in Weekly Sales', title_x=0.35)

        else:
            df2 = top_sales.top_dept_sales(
                f"day='{day2}' and month='{month2}' and year='{year2}' and store='{store2}'")

            plotly.vertical_bar_chart(df=df2, x='dept_x', y='Weekly Sales', text='Weekly Sales',
                                      color='#5cb85c', title='Top Departments in Weekly Sales', title_x=0.35)



elif option == 'Comparison':

        day1, month1, year1, store1, dept1, day2, month2, year2, store2, dept2 = comparison.manual_filter_options()
        add_vertical_space(3)

        df1 = comparison.sql(f"""day='{day1}' and month='{month1}' and year='{year1}' and 
                                 store='{store1}' and dept='{dept1}'""")

        df2 = comparison.sql(f"""day='{day2}' and month='{month2}' and year='{year2}' and 
                                 store='{store2}' and dept='{dept2}'""")

        comparison.manual_comparison(df1, df2)




elif option == 'Prediction':

    weekly_sales = prediction.predict_weekly_sales()

    if weekly_sales:

        # apply custom css style for prediction text
        style_prediction()

        st.markdown(f'### <div class="center-text">Predicted Sales = {weekly_sales}</div>', 
                    unsafe_allow_html=True)

        st.balloons()



elif option == 'Exit':
    
    add_vertical_space(2)

    col1,col2,col3 = st.columns([0.20,0.60,0.20])

    with col2:

        st.success('#### Thank you for your time. Exiting the application')
        st.balloons()

