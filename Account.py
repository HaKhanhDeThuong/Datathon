import streamlit as st
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin import auth

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