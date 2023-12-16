import streamlit as st 
import requests 
from streamlit_lottie import st_lottie
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from streamlit_extras.stylable_container import stylable_container
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import auth

st.set_page_config(page_title="WinX", page_icon="images/logo.png",layout="wide")

lottie_coding = "https://lottie.host/c3daf2d8-8561-41e5-b5b6-31e17731f227/2eCpn4kWuo.json"
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json

def header(url):
    st.markdown(f'<p style="background-color:#4F46E5;font-size:24px;border-radius:2%; padding: 5px; font-weight: bold">{url}</p>', unsafe_allow_html=True)

background_url = "images/background.jpg"
background_style = f"""
    <style>
        .st-d1 {{
            background-image: url('{background_url}');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center center;
            height: 100vh;  /* Adjust the height as needed */
        }}
    </style>
"""
##Header
with st.container():
        st.subheader("Personalized window displays drive sales and customer engagement.")
        st.title(":blue[Window displays that change with you]")
        st.write("We are the future of window displays. Our real-time solutions use AI to understand your customers and create personalized, engaging experiences that will keep them coming back for more.")
        st.button("Get a demo")

##Description
with st.container(): 
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        header("What can we do?")
        st.write("##")
        st.write(
    """
    **:blue[WinX: Stop window shoppers, start converting them.]**

    - **AI-powered displays**: Turn static windows into dynamic, personalized experiences.

    - **Smarter marketing**: Target ads & discounts to each shopper in real-time.

    - **Boost conversions**: Optimize displays based on what works best.

    """
        )
        left2, right2 = st.columns(2)
        with left2:
            result = st.button("Try WinX")
        with right2:
             result2 = st.button("Schedule a demo")

    with right_column:
        st_lottie(lottie_coding, height=300, key="analysis")


##Video Analytics
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        header("Video Analytics Available From Window Mirror")
        st.write("##")
        st.write(
    """
   Unlock business insights, optimize your marketing.

- **Video Analytics**: Dive deep into user engagement with heatmaps & data.

- **Website Heatmaps**: See where visitors click & scroll in real-time.

- **Dynamic Window Displays**: Captivate window shoppers with personalized promos.

- **Expert Marketing Consult**: Get the guidance you need to thrive.
    """)
    with right_column:
        st.image('images/video.png', width=800, caption="Four services of Video Analysis")        


##Service Utilization Process
with st.container():
    st.write("---")
    header("Our Service Utilization Process")
    st.write("##")
    image_col, text_col = st.columns((1,1))
    
    with image_col:
        st.write("##")
        st.image('images/service.png', width=600, caption="Service cycle")        
    with text_col:
        st.write("""
        **Win customers with every window glance.**

:blue[WinX] uses AI-powered cameras to turn your static window displays into dynamic, personalized shopping experiences. As shoppers pass by, WinX:
            
- **Identifies demographics**: Age, gender, and even what they’re carrying, all in real-time and without storing data.

- **Tailors ad content**: WinX handpicks the perfect ad from your favorite brands, ensuring each shopper sees something they’ll love.

- **Updates your displays**: Your window transforms instantly, showcasing the chosen ad with stunning visuals and irresistible offers.


""")


    ##Contact
    with st.container():
        st.write("---")
        header("Let's start solving your problems")
        st.write("##")
        st.markdown("Please provide your company information below to facilitate personalized service and enhance your experience.")

        conn = st.connection("gsheets", type=GSheetsConnection)
        
        #Fetch
        
        existing_data = conn.read(worksheet="Vendors", usecols=list(range(7)), ttl=7)
        existing_data = existing_data.dropna(how="all")
        
#List of Business Types and Products
PRODUCTS = [
    "Apparel", 
    "Jewelry", 
    "Housewares", 
    "Small appliances", 
    "Electronics", 
    "Groceries", 
    "Pharmaceutical Products"

]

# Onboarding New Vendor Form
with st.form(key="vendor_form"):
    company_name = st.text_input(label="Company Name*")
    phone = st.text_input(label="Phone*")
    email = st.text_input(label="Email*")
    products = st.multiselect("Products Offered", options=PRODUCTS)
    years_in_business = st.slider("Years in Business", 0, 50, 5)
    onboarding_date = st.date_input(label="Onboarding Date")
    additional_info = st.text_area(label="Additional Notes")

    # Mark mandatory fields
    st.markdown("**required*")

    submit_button = st.form_submit_button(label="Submit Your Information")

    # If the submit button is pressed
    if submit_button:
        # Check if all mandatory fields are filled
    
        if 'CompanyName' in existing_data.columns and existing_data["CompanyName"].str.contains(company_name).any():
            st.warning("This company name is already existed.")
            st.stop()
        else:
            # Create a new row of vendor data
            vendor_data = pd.DataFrame(
                [
                    {
                        "CompanyName": company_name,
                        "Products": ", ".join(products),
                        "Phone": phone,
                        "Email": email,
                        "YearsInBusiness": years_in_business,
                        "OnboardingDate": onboarding_date.strftime("%Y-%m-%d"),
                        "AdditionalInfo": additional_info,
                    }
                ]
            )

            # Add the new vendor data to the existing data
            updated_df = pd.concat([existing_data, vendor_data], ignore_index=True)

            # Update Google Sheets with the new vendor data
            conn.update(worksheet="Vendors", data=updated_df)

            st.success("Successfully submitted!")
            
##Signin/Login
if not firebase_admin._apps:
    cred = credentials.Certificate('datathon-ceb28-28c5a321ec51.json') 
    default_app = firebase_admin.initialize_app(cred)
    
def app():
# Usernm = []
    st.title('Log in :blue[WinX]')
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'useremail' not in st.session_state:
        st.session_state.useremail = ''



    def f(): 
        try:
            user = auth.get_user_by_email(email)
            print(user.uid)
            st.session_state.username = user.uid
            st.session_state.useremail = user.email
            
            global Usernm
            Usernm=(user.uid)
            
            st.session_state.signedout = True
            st.session_state.signout = True   
            st.balloons()
                       
        except: 
            st.warning('Login Failed')

    def t():
        st.session_state.signout = False
        st.session_state.signedout = False   
        st.session_state.username = ''


        
    
        
    if "signedout"  not in st.session_state:
        st.session_state["signedout"] = False
    if 'signout' not in st.session_state:
        st.session_state['signout'] = False    
        

        
    
    if not st.session_state["signedout"]:
    # only show if the state is False, hence the button has never been clicked
        st.markdown("Log in with the account we provided you via email.")
        email = st.text_input('Email Address')
        password = st.text_input('Password', type='password')
        st.button('Login', on_click=f)
            
            
    if st.session_state.signout:
                st.header('Your information:')
                st.text('Name: '+st.session_state.username)
                st.text('Email: '+st.session_state.useremail)
                st.button('Sign out', on_click=t) 
            
                
    

                            
    def ap():
        st.write('Posts')
if __name__ == "__main__":
    app()