import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pickle



#set up the sidebar with optionmenu
selected = option_menu(
    menu_title="Industrial Copper Modeling",      
    options=["Prediction", "Details"],  
    orientation="horizontal", 
        default_index=1 
)

#user input values for selectbox and encoded for respective features
class option():
    
    
    
    item_type_values=['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']

    country_values=[ 25.,  26.,  27.,  28.,  30.,  32.,  38.,  39.,  40.,  77.,  78., 79.,  80.,  84.,  89., 107., 113.]

    item_type_encoded = {'W':5.0, 'WI':6.0, 'S':3.0, 'Others':1.0, 'PL':2.0, 'IPL':0.0, 'SLAWR':4.0}

    status_values=['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM','Wonderful', 'Revised',
            'Offered', 'Offerable']

    application_values=[2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0,
                41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
    
    status_encoded = {'Lost':0, 'Won':1, 'Draft':2, 'To be approved':3, 'Not lost for AM':4,'Wonderful':5, 'Revised':6,
                    'Offered':7, 'Offerable':8}
    
    product_ref_values=[611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407,
                164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642,
                1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026,
                1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]
    customer_id=[30156308,30202938,30153963,30349574,30211560,30209509,30342192,30341428,30165529,30202362,30271717,30329143,
                 30153510,30211222,30152417,30406347,30162161,30198267,30336279,30268369,30357129,30197978,30394817,30329913,
                 30213691,30234602,30394116,30198682,30198586,30197494,30223403,30165992,30209745,30148135,30205658,30217607,
                 30221607,30202869,30205376,30202773,30214872,30403368,30268176,30161656,30280961,30332693,30218233,30397926,
                 30284855,30205825,30201369,30218009,30407192,30267018,30218442,30149080,30282890,30330352,30344725,30271120,
                 30341432,30205312,30160005,30225445,30198676,30406361,30235883,30357143,30199280,30267349,30329989,30166502,
                 30164472,30199269,30198577,30198507,30197899,30197989,30226969,30299911,30406343,30209675,30157111,30153920,
                 30341127,30297329,30338548,30272155,30148806,30295541,30349625,30161856,30349114,30199179,30275329,30213626,
                 30346730,30289108,30333168,30217988,30235913,30201846,30149089,30211048,30164616,30196997,30203386,30332631,
                 30393883,30271289,30210087,30199168,30272192,30197770,30203192,30202645,30164464,30206203,30201860,30202778,
                 30403366,30196870,30155824,30299962,30267637,30153509,30268729,30201560,30332186,30341940,30402346,30354197,
                 30197204,30328272,30340816,30217269,30402024,30166496,30299398,30211350,30164695,30198807,30164673,30197559,
                 30283915,30217504,30235584,30407177,30160683,30197144,30401985,30268656,30154020,30328008,30299929,30197271,
                 30403335,30271174,30292678,30393641,30272666,30165751,30227334,30406438,30329591,30402969,30199065,30198485,
                 30198761,30203120,30206229,30205973,30227447,30394336,30230920,30205061,30162074,30201745,30202053,30234009,
                 30210032,30333141,30164261,30287258,30280080,30198826,30271448,30332960,30202105,30356624,30164209,30407721,
                 30148496,30164630,30196886,30158293,30206167,30402201,30350564,30201881,30214705,30209814,30226059,30393881,
                 30160001,30201547,30148347,30161289,30197813,30197482,30156496,30205190,30299270,30337944,30218244,30160378,
                 30223382,30349338,30147722,30295605,30342177,30270857,30342017,30209521,30398176,30341656,30344971,30350826,
                 30205832,30353798,30354128,30329318,30199048,30165449,30398148,30229786,30219153,30158107,30165489,30406632,
                 30164377,30292440,30341590,30206744,30199273,30223158,30211558,30333845,30201046,30148629,30295600,30155929,
                 30394378,30272487,30230763,30231860,30214640,30198808,30217604,30341688,30203270,30330089,30272132,30201386,
                 30197225,30201113,30201589,30152592,30300422,30354180,30223867,30299349,30402535,30201478,30198298,30300452,
                 30202361,30328875,30222741,30272745,30222854,30278678,30198657,30164235,30198007,30398934,30403047,30198408,
                 30301062,30288168,30201735,30292464,30295947,30206401,30405904,30157776,30345783,30161189,30202517,30160375,
                 30268215,30230331,30279818,30346534,30291184,30353610,30288390,30205384,30405913,30280090,30338528,30353979,
                 30280641,30293493,30266629,30336052,30350038,30284310,30201336,30223670,30354198,30203366,30218904,30199259,
                 30206512,30148084,30356969,30267449,30356715,30267337,30336720,30295488,30148743,30394008,30344422,30272233,
                 30200854,30207014,30267173,30301129,30394631,30271445,30272234,30332676,30153956,30202870,30295830,30205242,
                 30300649,30397849,30334169,30291961,30211296,30393574,30148822,30161540,30332305,30350235,30209681,30230775,
                 30205728,30165984,30164100,30234084,30198820,30354376,30227497,30201912,30209173,30158105,30332472,30268901,
                 30356697,30165222,30357257,30198074,30210704,30341617,30398153,30299670,30333217,30295288,30292369,30161088,
                 30197265,30235451,30344937,30282789,30205888,30407380,30272500,30332976,30288101,30162145,30267707,30341512,
                 30197397,30346185,30229766,30148907,30344408,30218709,30161350,30213686,30403578,30266897,30197000,30165703,
                 30300202,30272768,30223319,30166521,30329993,30401748,30267268,30407624,30330056,30235865,30280155,30157862,
                 30399766,30354186,30336971,30332041,30202260,30280697,30198841,30394106,30282773,30202501,30344219,30271881,
                 30291680,30279382,30283830,30204967,30287088,30399988,30288392,30288377,30222746,30149413,30214647,30344235,
                 30284064,30148826,30162405,30210549,30271497,30394328,30275723,30345494,30203029,30285248,30271380,30150133,
                 30282753,30276619,30267627,30354136,30155835,30301590,30272289,30227718,30161943,30157952,30352779,30282949,
                 30202665,30235401,30264704,30340120,30217368,30397719,30272262,30210295,30217974,30345684,30278792,30354217,
                 30353306,30356538,30342118,30163992,30271764,30205568,30407558,30197785,30221720,30164965,30226992,30283495,
                 30341927,30394353,30287099,30271892,30333936,30211030,30197496,30349328,30295498,30206504,30407482,30285306,
                 30230583,30196884,30197193,30334215,30205078,30349000,30227654,30397446,30205929,30406628,30281163,30267017,
                 30286858,30159894,30266602,30333322,30280196,30202633,30350630,30292171,30217445,30219767,30296071,30354487,
                 30155782,30341417,30328266,30341380,30209332,30233801,30395094,30201914,30276343,30197529,30297082,30276344,
                 30160788,30393833,30354200,30206752,30292608,30288790,30230560,30350017,30342417,30276326,30230282,30221960,
                 30197028,30275200,30337991,30348538,30205638,30205704,30202916,30399897,30329536,30199248,30406442,30209578,
                 30406116,30300710,30203105,30156261,30205174,30164696,30332474,30264793,30164728,30147848,30201911,30156842,
                 30279705,30206677,30271798,30341568,30201223,30202369,30205434,30218006,30406649,30357385,30149105,30281159,
                 30272650,30205963,30231018,30279156,30197457,30229528,30338176,30271444,30164964,30165029,30334154,30270976,
                 30198423,30333056,30161450,30201748,30354218,30157104,30394057,30218705,30202391,30341921,30341671,30394097,
                 30290964,30200964,30333962,30148586,30267125,30198328,30402585,30346629,30350566,30398981,30231432,30403562,
                 30407184,30395649,30407168,30341856,30205489,30408185,30407456,30271383,30296635,30225461,30209985,30297111,
                 30287910,30354480,30348747,30154025,30284251,30342537,30234070,30292278,30147620,30275784,30206182,30336057,
                 30202856,30357253,30403563,30166282,30394913,30219402,30346201,30297495,30196960,30349303,30197464,30209290,
                 30283754,30201322,30197392,30149530,30206006,30280416,30291449,30202168,30267668,30218033,30148854,30210839,
                 30207425,30266641,30349352,30197382,30394753,30153176,30205846,30402792,30160907,30165526,30197636,30394923,
                 30210971,30288891,30402054,30357496,30288326,30285108,30395031,30147802,30270762,30164779,30397927,30301321,
                 30160373,30272260,30209830,30267096,30406346,30398230,30162304,30162122,30225652,30337802,30271284,30403536,
                 30160064,30203360,30266859,30225641,30148901,30398170,30232058,30394585,30267441,30267081,30202377,30217497,
                 30327942,30199195,30357142,30284689,30157092,30275216,30345717,30357498,30201633,30332881,30165716,30201866,
                 30356954,30272821,30272395,30205952,30357481,30227513,30399734,30342279,30349579,30166011,30284057,30205845,
                 30202102,30272198,30205681,30205321,30205158,30336761,30201744,30234136,30211047,30214651,30399736,30197016,
                 30225640,30147616,30267177,30356790,30272503,30268353,30279672,30293300,30402971,30406151,30234625,30407307,
                 30276505,30203273,30292762,30398091,30346297,30292441,30230922,30201146,30352577,30357456,30207233,30205847,
                 30201848,30356683,30203189,30165508,30201096]

#set up information for the 'get prediction' menu
if selected == 'Prediction':
    title_text = '''<h1 style='font-size: 32px;text-align: center;color:white;'>Selling Price of Copper and Status Prediction </h1>'''
    st.markdown(title_text, unsafe_allow_html=True)
    
    #set up option menu for selling price and status menu
    select=option_menu('',options=["Selling Price","Status"])


    if select == 'Selling Price':
        st.markdown("<h5 style=color:white>Provide the below required details:",unsafe_allow_html=True)
        st.write('')

        # creted form to get the user input 
        with st.form('prediction'):
            col1,col2=st.columns(2)
            with col1:

                customer=st.selectbox(label='Customer ID',options=option.customer_id,index=None)

                item_type=st.selectbox(label='Item Type',options=option.item_type_values,index=None)

                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                application=st.selectbox(label='Application',options=option.application_values,index=None)

                quantity=st.number_input(label='Quantity',min_value=0.1)

                thickness=st.number_input(label='Thickness',min_value=0.1)

            with col2:

                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                country=st.selectbox(label='Country',options=option.country_values,index=None)

                product_ref=st.selectbox(label='Product Ref',options=option.product_ref_values,index=None)

                width=st.number_input(label='Width',min_value=1.0)

                status=st.selectbox(label='Status',options=option.status_values,index=None)


                st.markdown('<br>', unsafe_allow_html=True)
                
                button=st.form_submit_button('PREDICT',use_container_width=True)

        if button:
            #check whether user fill all required fields
            if not all([item_date, delivery_date, country, item_type, application, product_ref,
                        customer, status, quantity, width, thickness]):
                st.error("Fill all the required fields.")

            else:
                
                #opened pickle model and predict the selling price with user data
                with open('/Users/rathikakn/Desktop/Guvi/Project4coppermodeling/Regressormodelfile.pkl','rb') as files:
                    predict_model=pickle.load(files)

                # customize the user data to fit the feature 
                status=option.status_encoded[status]
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)

                #predict the selling price with regressor model
                user_data=np.array([[customer, country, status, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log ]])
                
                pred=predict_model.predict(user_data)

                selling_price=np.exp(pred[0])

                #display the predicted selling price 
                st.subheader(f" Selling Price : {selling_price:.2f}") 

    if select == 'Status':
        st.markdown("<h5 style=color:grey;>Provide the below required details:",unsafe_allow_html=True)
        st.write('')

        #creted form to get the user input 
        with st.form('classifier'):
            col1,col2=st.columns(2)
            with col1:

                customer=st.selectbox(label='Customer ID',options=option.customer_id,index=None)

                item_date=st.date_input(label='Item Date',format='DD/MM/YYYY')

                item_type=st.selectbox(label='Item Type',options=option.item_type_values,index=None)

                quantity=st.number_input(label='Quantity',min_value=0.1)

                application=st.selectbox(label='Application',options=option.application_values,index=None)

                thickness=st.number_input(label='Thickness',min_value=0.1)

            with col2:

                delivery_date=st.date_input(label='Delivery Date',format='DD/MM/YYYY')

                country=st.selectbox(label='Country',options=option.country_values,index=None)

                product_ref=st.selectbox(label='Product Ref',options=option.product_ref_values,index=None)

                width=st.number_input(label='Width',min_value=1.0)

                selling_price=st.number_input(label='Selling Price',min_value=0.1)

                st.markdown('<br>', unsafe_allow_html=True)
                
                button=st.form_submit_button('PREDICT',use_container_width=True)

        if button:
            #check whether user fill all required fields
            if not all([item_date, delivery_date, country, item_type, application, product_ref,
                        customer,quantity, width, thickness,selling_price]):
                st.error("Fill all the required fields.")

            else:
                #opened pickle model and predict status with user data
                with open('/Users/rathikakn/Desktop/Guvi/Project4coppermodeling/Classifiermodelfile.pkl','rb') as files:
                    model=pickle.load(files)

                # customize the user data to fit the feature 
                item_type=option.item_type_encoded[item_type]

                delivery_time_taken=abs((item_date - delivery_date).days)

                quantity_log=np.log(quantity)
                thickness_log=np.log(thickness)
                selling_price_log=np.log(selling_price)

                #predict the status with classifier model
                user_data=np.array([[customer, country, item_type ,application, width, product_ref,
                                    delivery_time_taken, quantity_log, thickness_log, selling_price_log ]])
                
                status=model.predict(user_data)

                #display the predicted status 
                if status==1:
                    st.subheader(f"Status : Won")

                else:
                    st.subheader(f"Status : Lost")


#set up information for 'About' menu 
if selected == "Details":
    st.subheader(':red[Project Title :]')
    st.markdown('<h5> Industrial Copper Modeling',unsafe_allow_html=True)

    st.subheader(':red[Domain :]')
    st.markdown('<h5> Manufacturing ',unsafe_allow_html=True)

    st.subheader(':red[Skills & Technologies :]')
    st.markdown('<h5> Python scripting, Data Preprocessing, EDA, Streamlit, Machine learning ',unsafe_allow_html=True)

    st.subheader(':red[Overview :]')
    st.markdown('<h5> Data Understanding:',unsafe_allow_html=True)
    st.markdown('''<li>Identify the types of variables (continuous, categorical) and their distributions. Some rubbish values are present in ‘Material_Reference’ which starts with ‘00000’ value which should be converted into null. Treat reference columns as categorical variables. INDEX may not be useful.''',unsafe_allow_html=True)
    
    st.markdown('<h5>Data Preprocessing:',unsafe_allow_html=True)
    st.markdown('''<li>Handle missing values with mean/median/mode.
                <li>Treat Outliers using IQR or Isolation Forest from sklearn library.
                <li>Identify Skewness in the dataset and treat skewness with appropriate data transformations, such as log transformation(which is best suited to transform target variable-train, predict and then reverse transform it back to original scale eg:dollars), boxcox transformation, or other techniques, to handle high skewness in continuous variables.
                <li>Encode categorical variables using suitable techniques, such as one-hot encoding, label encoding, or ordinal encoding, based on their nature and relationship with the target variable.
                ''',unsafe_allow_html=True)
    
    st.markdown('<h5>Exploratory Data Analysis(EDA):',unsafe_allow_html=True)
    st.markdown('''<li>Try visualizing outliers and skewness(before and after treating skewness) using Seaborn.''',unsafe_allow_html=True)
    
    
    st.markdown('<h5>Feature Engineering:',unsafe_allow_html=True) 
    st.markdown('''<li>Engineer new features if applicable, such as aggregating or transforming existing features to create more informative representations of the data. And drop highly correlated columns using SNS HEATMAP.''',unsafe_allow_html=True)
    
    st.markdown('<h5>Model Building and Evaluation:',unsafe_allow_html=True)
    st.markdown('''<li>Split the dataset into training and testing/validation sets. 
                    <li>Train and evaluate different classification models, such as ExtraTreesClassifier, XGBClassifier, or Logistic Regression, using appropriate evaluation metrics such as accuracy, precision, recall, F1 score, and AUC curve. 
                    <li>Optimize model hyperparameters using techniques such as cross-validation and grid search to find the best-performing model.
                    <li>Interpret the model results and assess its performance based on the defined problem statement.
                    <li>Same steps for Regression modelling.(note: dataset contains more noise and linearity between independent variables so itll perform well only with tree based models)
''',unsafe_allow_html=True)
    
    st.markdown('<h5>Model GUI: ',unsafe_allow_html=True)
    st.markdown('''Using streamlit module, created interactive page with:    
                    <li>task input( Regression or Classification) and 
                    <li>created an input field where you can enter each column value except ‘Selling_Price’ for regression model and  except ‘Status’ for classification model. 
                    <li>performed the same feature engineering, scaling factors, log/any transformation steps which you used for training ml model and predict this new data from streamlit and display the output.
''',unsafe_allow_html=True)
    st.markdown('''''',unsafe_allow_html=True)