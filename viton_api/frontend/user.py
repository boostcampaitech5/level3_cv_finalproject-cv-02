# streamlit
import streamlit as st
# import streamlit_authenticator as stauth

# built-in library
import yaml
from yaml.loader import SafeLoader


class User:
    def __init__(self):
        self.root = "/opt/ml/storage/accounts.yaml"
        
        with open(self.root) as f:
            self.config = yaml.load(f, Loader=SafeLoader)
        
    
    def create_account(self):
        email = st.text_input("이메일: ")
        name = st.text_input("이름: ")
        candidate = st.text_input("아이디: ")
        
        progress = False
        if candidate:
            if candidate not in self.config["credentials"]["usernames"].keys():
                st.text("사용 가능한 아이디입니다!")
                progress = True
            else:
                st.text("중복된 아이디입니다. 다시 입력해주세요.")
        
        if progress:
            password = st.text_input("비밀번호: ")
            
            if password:
                hashed_password = stauth.Hasher([password]).generate()
                self.config['credentials']['usernames'][candidate] = {'email': email,
                                                                      'name': name,
                                                                      'password': hashed_password}

                with open(self.root, 'w') as f:
                    yaml.safe_dump(self.config, f, allow_unicode=True)
                
                st.text("회원 가입이 완료되었습니다!")
    

    # TODO: 로그인 기능 만들기
    def login(self):
        authenticator = stauth.Authenticate(
            self.config['credentials'],
            self.config['cookie']['name'],
            self.config['cookie']['key'],
            self.config['cookie']['expiry_days'],
            self.config['preauthorized']
        )

        name, authentication_status, username = authenticator.login("Login", 'main')
        
        if authentication_status:
            authenticator.logout('Logout', 'main', key='unique_key')
            st.write(f'Welcome *{name}*')
            st.title('Some content')
        elif authentication_status is False:
            st.error('Username/password is incorrect')
        elif authentication_status is None:
            st.warning('Please enter your username and password')


# Test Code
if __name__ == "__main__":
    user = User()
    user.create_account()
    user.login()