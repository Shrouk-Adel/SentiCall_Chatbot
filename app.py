import streamlit as st
import torch
import joblib
import re
from transformers import AutoTokenizer, AutoModel, pipeline
from transformers import pipeline
from nltk.corpus import stopwords
from qalsadi.lemmatizer import Lemmatizer
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
import nltk
import pandas as pd
from streamlit_chat import message
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import plotly.express as px
import warnings
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Configure Google Generative AI with a valid API key
genai.configure(api_key='your_API_Key')

# Initialize chat session only once
@st.cache_resource
def initialize_chat():
    return genai.GenerativeModel('gemini-1.5-pro').start_chat()

# Initialize the chat session
chat = initialize_chat()

# Load models and tokenizer only once
@st.cache_resource
def load_models():
    vectorizer = joblib.load('models/vectorizer.joblib')
    logistic_model = joblib.load('models/logistic_model.joblib')
    tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-mini-arabic")
    bert_model = CustomBertClassifier()
    bert_model.load_state_dict(torch.load('models/bert_classifier.pth'), strict=False)
    bert_model.eval()
    return vectorizer, logistic_model, tokenizer, bert_model

# Load MARBERT model pipeline
@st.cache_resource
def load_marbert_pipeline():
    return pipeline('text-classification', model='Ammar-alhaj-ali/arabic-MARBERT-sentiment')

# Custom BERT Classifier
class CustomBertClassifier(torch.nn.Module):
    def __init__(self, num_classes=3, model_name="asafaya/bert-mini-arabic"):
        super(CustomBertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.new_classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.new_classifier(cls_embeddings)

# Sentiment prediction and text preprocessing functions
def return_clean_text(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words("arabic"))
    emojis = {"🙂": "يبتسم", "😂": "يضحك", "💔": "قلب حزين", "❤️": "حب", "😭": "يبكي", "😢": "حزن", "😔": "حزن", "😄": "يضحك"}
    text = re.sub(r'[^\w\s]', ' ', text)
    text = ''.join([emojis.get(char, char) for char in text])
    return " ".join([word for word in Lemmatizer().lemmatize_text(text) if word not in stop_words])

def predict_sentiment(text):
    vectorizer, logistic_model, tokenizer, bert_model = load_models()
    clean_text = return_clean_text(text)
    tokenized_text = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        bert_output = bert_model(tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"])
        bert_pred = torch.argmax(bert_output, dim=1).item()
    return {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😊"}[bert_pred]

def predict_sentiment_with_marbert(text):
    model = load_marbert_pipeline()
    clean_text = return_clean_text(text)
    result = model(clean_text)
    #sentiment = result[0]['label']
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']
    #return sentiment
    # Determine the sentiment emoji based on the label
    if sentiment_label == 'LABEL_0':  # negative sentiment
        emoji = '😞'
        sentiment_text = "negative"
    elif sentiment_label == 'LABEL_1':  # positive sentiment
        emoji = '😊'
        sentiment_text = "positive"
    else:
        emoji = '😐'
        sentiment_text = "neutral"

    # Form a cute and friendly output
    return f"Sentiment: {sentiment_text} {emoji}\nScore: {sentiment_score * 100:.2f}%"


def ask_question(chat, conversation, question):
    prompt = f"{conversation}\n\nسؤال: {question}"
    response = chat.send_message(prompt)
    return response.text if response else "No response available."

# Login Page
def login_page():
    st.title("Arabic Sentiment Analysis & Chatbot")
    st.markdown("""
    <style>
    /* Set black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Title and text styling */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
        animation: fadeIn 1s ease-in;
    }

    /* Loading spinner styling */
    .stSpinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        animation: spin 1s linear infinite;
    }

    /* Transcription and response box animation */
    .stTextArea, .stTextInput, .stButton {
        transition: all 0.3s ease;
    }
    .stTextArea:hover, .stTextInput:hover, .stButton:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.5);
    }



    /* Fade-in animation for all elements */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Spinner animation for loading indicator */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

    # CSS styling to modify the button appearance
    st.markdown("""
    <style>
    /* Set a global black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Style for all button elements */
    .stButton > button {
        color: white !important; /* Ensures button text is white */
        font-size: 18px !important; /* Increases font size for better readability */
        font-weight: bold !important; /* Makes text bold */
        background-color: #333 !important; /* Dark button background */
        border-radius: 8px; /* Rounded corners for button */
        padding: 10px 20px; /* Adds padding to look like a custom button */
        transition: background-color 0.3s, transform 0.3s;
    }

    /* Hover effect for the button */
    .stButton > button:hover {
        background-color: #555 !important; /* Slightly lighter background on hover */
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3); /* Adds a shadow on hover */
        transform: scale(1.05); /* Slight zoom-in effect on hover */
    }
    </style>
""", unsafe_allow_html=True)


    st.title("Login")
    username = st.text_input("Username or Email Address", value="")
    password = st.text_input("Password", type="password", value="")

    if st.button("Log In"):
        if username == "Mohamed Ammar" and password == "0000":
            st.session_state['logged_in'] = True
            st.session_state.page = 'main'
        else:
            st.error("Invalid username or password")
def about_us_page():
    # Language selection
    language = st.radio("Select Language", ("English", "Arabic"))

    if language == "English":
        st.title("📖 About Us")
        st.write("""
            Welcome to the **Arabic Sentiment Analysis and Customer Dataset Chatbot Application**!

            This application is designed to help you:

            1. **Analyze the sentiment of Arabic text** extracted from audio files.
            2. **Interact with a customer dataset chatbot** to get answers to customer-related queries.
            3. **Visualize sentiment analysis results** and chatbot interactions through dynamic charts.

            ## Features of the Application:

            ### 1. **Sentiment Analysis Page**:
            - This page allows you to upload Arabic audio files (in `.mp3` or `.mpeg` format).
            - The audio file is transcribed into text, and then the sentiment of the text is analyzed using a pre-trained BERT-based model.
            - The sentiment analysis results categorize the text as either **Positive**, **Neutral**, or **Negative**.
            - Additionally, the system provides an option to interact with a chatbot, asking questions based on the transcribed text. The chatbot responds with answers derived from a customer dataset.

            ### 2. **Customer Dataset Chat Page**:
            - On this page, you can chat with a customer dataset chatbot.
            - The chatbot uses a CSV file containing customer queries and responses, enabling it to provide intelligent answers based on this data.
            - The chatbot is integrated with Google Generative AI, allowing it to understand and generate human-like responses to your questions.
            - You can also ask questions related to the customer dataset, and the chatbot will provide answers directly from the database.

            ### 3. **About Us Page**:
            - This page provides an overview of the project, including its purpose, the technologies used, and the benefits of using this application.
            - You will find details on how sentiment analysis works, the chatbot’s capabilities, and how this tool can help in analyzing customer interactions and improving user experiences.

            ## How to Use:
            1. **Login** to the application to access the full features.
            2. **Sentiment Analysis**: Upload an Arabic audio file to get its sentiment score and interact with the chatbot.
            3. **Customer Dataset Chat**: Ask questions related to the customer dataset, and get real-time responses from the chatbot and Visualize sentiment analysis results and chatbot interactions through dynamic charts.

            ## Technologies Used:
            - **Natural Language Processing (NLP)** for sentiment analysis using BERT models.
            - **Speech Recognition** to transcribe Arabic audio files.
            - **Google Generative AI** to power the chatbot functionality.
            - **Streamlit** for the interactive web application.

            We hope this tool helps you with your sentiment analysis tasks and enhances your customer support or analysis processes!

            If you have any questions or need assistance, feel free to reach out via the contact page or through the chat.
        """)

    else:  # Arabic

        st.title("📖 من نحن")

        st.write("""
        !مرحباً بكم في **تطبيق تحليل المشاعر باللغة العربية ودردشة مع قاعدة بيانات العملاء**

        -:  تم تصميم هذا التطبيق لمساعدتك في

        - **تحليل المشاعر للنصوص العربية** المستخلصة من ملفات الصوت.
        - **التفاعل مع روبوت دردشة قائم على بيانات العملاء** للحصول على إجابات لأسئلة متعلقة بالعملاء.
        - **عرض نتائج تحليل المشاعر** والتفاعلات مع روبوت الدردشة من خلال الرسوم البيانية التفاعلية.

        ## مميزات التطبيق

        ####  **صفحة تحليل المشاعر**
        - تتيح لك هذه الصفحة تحميل ملفات الصوت باللغة العربية (بصيغة `.mp3` أو `.mpeg`).
        - يتم تحويل الصوت إلى نص، ثم يتم تحليل المشاعر باستخدام نموذج BERT المدرب مسبقًا.
        - تصنف نتائج تحليل المشاعر النصوص إلى **إيجابي** أو **محايد** أو **سلبي**.
        - بالإضافة إلى ذلك، يوفر النظام خيار التفاعل مع روبوت الدردشة، حيث يمكنك طرح أسئلة استنادًا إلى النص المحول. يرد الروبوت بإجابات مستمدة من قاعدة بيانات العملاء.

        #### **صفحة دردشة مع قاعدة بيانات العملاء**
        - في هذه الصفحة، يمكنك الدردشة مع روبوت دردشة لقاعدة بيانات العملاء.
        - يستخدم الروبوت ملف CSV يحتوي على استفسارات العملاء والإجابات، مما يتيح له تقديم إجابات ذكية استنادًا إلى البيانات.
        - يتم تكامل الروبوت مع **الذكاء الاصطناعي التوليدي من جوجل**، مما يسمح له بفهم الأسئلة وتوليد إجابات مشابهة للبشر.
        - يمكنك أيضًا طرح أسئلة حول قاعدة البيانات، وسيقدم الروبوت إجابات مباشرة من قاعدة البيانات.

        #### **صفحة من نحن**
        - توفر هذه الصفحة لمحة عامة عن المشروع، بما في ذلك الغرض منه والتقنيات المستخدمة وفوائد استخدام هذا التطبيق.
        - ستجد تفاصيل حول كيفية عمل تحليل المشاعر، وقدرات الروبوت، وكيف يمكن أن يساعد هذا الأداة في تحليل تفاعلات العملاء وتحسين التجارب.

        #### كيفية الاستخدام
        1. **تسجيل الدخول** للوصول إلى جميع الميزات.
        2. **تحليل المشاعر**: تحميل ملف صوتي باللغة العربية للحصول على نتيجة التحليل والتفاعل مع الروبوت.
        3. **دردشة مع قاعدة بيانات العملاء**: طرح أسئلة متعلقة بقاعدة بيانات العملاء، والحصول على إجابات فورية من الروبوت.
        4. **الرسوم البيانية**: عرض نتائج تحليل المشاعر وتفاعلات الروبوت من خلال الرسوم البيانية التفاعلية.

        #### التقنيات المستخدمة
        - **معالجة اللغة الطبيعية (NLP)** لتحليل المشاعر باستخدام نماذج BERT.
        - **التعرف على الصوت** لتحويل الملفات الصوتية العربية إلى نصوص.
        - **الذكاء الاصطناعي التوليدي من جوجل** لتمكين الروبوت.
        - **Streamlit** لتطوير التطبيق التفاعلي.

       ! نأمل أن يساعدك هذا التطبيق في مهام تحليل المشاعر وتعزيز دعم العملاء أو عمليات التحليل

         . إذا كان لديك أي أسئلة أو بحاجة إلى مساعدة، لا تتردد في التواصل معنا عبر صفحة الاتصال أو من خلال الدردشة
    """)


# Function to display the team members page
def team_members_page():
    st.title("👥Meet Our Team")

    team_members = [
        {
            "name": "Mahmoud Saad Mahmoud",
            "role": "Artificial Intelligence Engineer",
            "location": "Cairo, Egypt",
            "email": "mahmoud.saad.mahmoud.11@gmail.com",
            "phone": "(+20) 1145688541",

        },
        {
            "name": "Mohamed Ahmed Ammar",
            "role": "Machine Learning Engineer",
            "location": "Nasr City, Cairo",
            "email": "mohumedammar@gmail.com",
            "phone": "+20 1016231323",

        },
        {
            "name": "Shrouk Adel Mahmoud Mohamed",
            "role": "Junior Machine Learning Engineer",
            "location": "Ashmoun, Menofya, Egypt",
            "email": "Shrouk297adel@gmail.com",
            "phone": "(+2) 01204787684",
        },
        {
            "name": "Fatma Saeed Foaad",
            "role": "Machine Learning Engineer",
            "location": "Cairo, Egypt",
            "email": "fatmasaeedfoaad12@gmail.com",
            "phone": "01153892413",
        }

    ]

    # Display each team member with profile summary
    for member in team_members:
        st.subheader(f"{member['name']}")
        st.markdown(f"<small>{member['role']}</small>", unsafe_allow_html=True)
        st.write(f"📍 Location: {member['location']}")
        st.write(f"📧 Email: {member['email']}")
        st.write(f"📞 Phone: {member['phone']}")
        st.markdown("---")  # Horizontal separator

# Function for Data Analysis and Visualization
def dataset_analysis_page():
    st.title("Customer Dataset Analysis & Visualization")

    # Load Dataset
    csv_file_path = 'datasets/CompanyReviews.csv'
    df = pd.read_csv(csv_file_path)
    if df is not None:
            # Dataset Overview
            #st.subheader("Dataset Overview")
            st.write("Sample Data:")
            st.dataframe(df.head())

            # Create two columns
            col1, col2 = st.columns(2)
            with col1:
                st.write("Dataset Summary:")
                st.write(df.describe())

            with col2:
            # Class Balance Visualization (if applicable)
                if 'rating' in df.columns:
                    st.write("Class Balance")
                    st.write(df['rating'].value_counts())
                    class_names = ['negative', 'neutral', 'positive']
                    # st.write("Dataset Summary:")
                    # st.write(df.describe())

            # Class Balance Visualization (if applicable)
            if 'rating' in df.columns:
                # Plot Class Distribution
                st.write("### Class Distribution")
                plt.figure(figsize=(8, 6))
                sns.countplot(data=df, x='rating', palette='viridis')
                ax = plt.gca()
                ax.set_xticklabels(class_names)
                plt.title('Class Distribution')
                st.pyplot(plt)

                # Create the pie chart
                # Streamlit app setup
                st.title("Ratings Pie Chart")
                fig = go.Figure(data=[go.Pie(labels=["postive","negative","neutral"],values=[df.rating[df.rating==x].count() for x in df.rating.unique()],pull=[0, 0.1, 0])])
                fig.update_layout(title="Ratings")
                # Display the chart in Streamlit
                st.plotly_chart(fig)


                st.title("company Pie Chart")
                fig=go.Figure(data=[go.Pie(labels=df.company.unique(),values=df.company.value_counts(),hole=0.5)])
                fig.update_layout(title="Ratings")
                st.plotly_chart(fig)


                st.title('Company Distribution')
                # Create the Seaborn plot
                plt.figure(figsize=(15, 8))
                sns.countplot(data=df, x='company', palette='viridis')
                plt.xlabel('Company')
                plt.ylabel('Count')
                plt.title('Company Distribution')
                # Display the plot in Streamlit
                st.pyplot(plt)

                st.title("Companies and Feedbacks")
                df=df.copy()
                df.rating=df.rating.map({0:"neutral",1:"postive",-1:"negative"})
                # Create the Sunburst chart
                fig=px.sunburst(df,path=["company","rating"],title="Companies and Feedbacks", color_continuous_scale='RdBu',color="rating")
                fig.update_traces(textinfo='label+percent parent')
                # Display the chart in Streamlit
                st.plotly_chart(fig)

                for companyName in df.company.unique():
                    fig = go.Figure(data=[go.Bar(y=df.rating[df["company"]==companyName].value_counts(),x=df.rating[df["company"]==companyName].unique())])
                    fig.update_layout(title=companyName+' Ratings')
                    st.plotly_chart(fig)





# Main app navigation
def main_app_page():

    with st.sidebar:

        st.markdown("""
    <style>
    /* Set gray background and light gray text */
    .main, .reportview-container {
        background-color: #2e2e2e; /* Dark gray background */
        color: #d3d3d3; /* Light gray text */
    }

    /* Title and text styling */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #ffffff !important; /* White text for better contrast */
        animation: fadeIn 1s ease-in; /* Fade-in animation */
    }

    /* Loading spinner styling */
    .stSpinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid #ffffff; /* White spinner */
        animation: spin 1s linear infinite; /* Spinner animation */
    }

    /* Input box, button hover animations */
    .stTextArea, .stTextInput, .stButton {
        transition: all 0.3s ease;
    }
    .stTextArea:hover, .stTextInput:hover, .stButton:hover {
        transform: scale(1.05); /* Slight zoom-in on hover */
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.5); /* Glow effect */
    }

    /* Fade-in animation for all elements */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Spinner animation for loading indicator */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

        # CSS styling to modify button appearance
        st.markdown("""
    <style>
    /* Style for all button elements */
    .stButton > button {
        color: #ffffff !important; /* Ensures button text is white */
        font-size: 18px !important; /* Larger text */
        font-weight: bold !important; /* Bold text */
        background-color: #4a4a4a !important; /* Medium gray button background */
        border-radius: 8px; /* Rounded corners */
        padding: 10px 20px; /* Add padding */
        transition: background-color 0.3s, transform 0.3s;
    }

    /* Hover effect for the button */
    .stButton > button:hover {
        background-color: #6a6a6a !important; /* Lighter gray on hover */
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3); /* Glow effect */
        transform: scale(1.05); /* Slight zoom-in */
    }
    </style>
    """, unsafe_allow_html=True)

        st.title("🤖 SentiCall")
        st.markdown(
        """
        *Analyze customer call sentiments*
        - Explore data through interactive chat
        - Generate charts and uncover insights
        - Improve service quality with actionable patterns
        - Supports multilingual queries for a global reach 🌍
        """
    )
        st.markdown("Made with ❤️ by *Your Team*")


    st.markdown("""
    <style>
    /* Set black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Title and text styling */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
        animation: fadeIn 1s ease-in;
    }

    /* Loading spinner styling */
    .stSpinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        animation: spin 1s linear infinite;
    }

    /* Transcription and response box animation */
    .stTextArea, .stTextInput, .stButton {
        transition: all 0.3s ease;
    }
    .stTextArea:hover, .stTextInput:hover, .stButton:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.5);
    }

    /* Fade-in animation for all elements */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Spinner animation for loading indicator */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

    # CSS styling to modify the button appearance
    st.markdown("""
    <style>
    /* Set a global black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Style for all button elements */
    .stButton > button {
        color: white !important; /* Ensures button text is white */
        font-size: 18px !important; /* Increases font size for better readability */
        font-weight: bold !important; /* Makes text bold */
        background-color: #333 !important; /* Dark button background */
        border-radius: 8px; /* Rounded corners for button */
        padding: 10px 20px; /* Adds padding to look like a custom button */
        transition: background-color 0.3s, transform 0.3s;
    }

    /* Hover effect for the button */
    .stButton > button:hover {
        background-color: #555 !important; /* Slightly lighter background on hover */
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3); /* Adds a shadow on hover */
        transform: scale(1.05); /* Slight zoom-in effect on hover */
    }
    </style>
""", unsafe_allow_html=True)

    st.markdown("""
    <style>
        /* Styling the overall page to have a colored top bar */
        .css-1lcbq3d {
            background-color: #2e8b57;  /* Set a green background color for the top bar */
            padding: 20px;
            font-size: 22px;  /* Increase font size of the tab text */
            font-weight: bold;
            color: white;
            text-align: center;
            border-radius: 10px 10px 0 0; /* Round the top corners */
        }

        /* Styling for tabs */
        .streamlit-tabs .tab {
            background-color: #3cb371;  /* Background color for tabs */
            color: white;
            font-size: 18px;  /* Increase font size of the tab text */
            padding: 15px 30px;  /* Increase padding to make the tabs larger */
            border-radius: 10px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

        /* Hover effect for tabs */
        .streamlit-tabs .tab:hover {
            background-color: #2e8b57;  /* Darker green when hovering */
            transform: scale(1.1);  /* Slightly enlarge the tab */
        }

        /* Active tab styling */
        .streamlit-tabs .tab.selected {
            background-color: #228b22;  /* Dark green for selected tab */
            color: yellow;  /* Change text color to yellow for selected tab */
        }

        /* Animation for the tab bar (you can modify this) */
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Apply the animation to the top bar */
        .css-1lcbq3d {
            animation: slideIn 0.5s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 ,tab5, tab6 = st.tabs(
        ["Sentiment Analysis and chatbot", "Customer Dataset Chat","Dataset Analysis and Visualization","Tableau Dashboard","About Us", "Team Members"]
    )

    # Main content area based on selected tab
    with tab1:
        sentiment_analysis_page()

    with tab2:
        #st.markdown("##### Customer Dataset Chat")
        customer_dataset_chat_page()
    with tab3:
        dataset_analysis_page()

    with tab4:
        st.markdown("# Tableau Dashboard")
        st.markdown("""
        <style>
        .dashboard-link {
            display: inline-block;
            color: white;
            background-color: #0073e6;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            font-size: 18px;
            font-weight: bold;
            transition: background-color 0.3s ease, transform 0.2s ease;
        }
        .dashboard-link:hover {
            background-color: #005bb5;
            transform: scale(1.05);
        }
        </style>
        <a href="https://public.tableau.com/app/profile/mohamed.ammar8125/viz/project_17320093001030/Dashboard1?publish=yes" target="_blank" class="dashboard-link">
            Open Tableau Dashboard
        </a>
        """, unsafe_allow_html=True)

    with tab5:
        #st.markdown("### About Us")
        st.markdown("Learn more about the app and its features.")
        about_us_page()

    with tab6:
        #st.markdown("### Meet Our Team")
        team_members_page()

# Sentiment Analysis Page
def sentiment_analysis_page():
    st.subheader("Sentiment Analysis from Audio")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "mpeg"])

    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    if uploaded_file is not None:
        with open("temp_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio = AudioSegment.from_file("temp_audio.mp3", format="mp3")
        audio.export("temp_audio.wav", format="wav")

        if st.button("Transcribe Audio to Text"):
            recognizer = sr.Recognizer()
            with sr.AudioFile("temp_audio.wav") as source:
                audio_data = recognizer.record(source)
            try:
                st.session_state.transcribed_text = recognizer.recognize_google(audio_data, language="ar-SA")
                st.write("Transcribed Text:")
                st.write(st.session_state.transcribed_text)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError:
                st.error("There was an error with the speech recognition service.")

    if st.session_state.transcribed_text:
        model_option = st.selectbox("Choose Sentiment Analysis Model", ["BERT (Custom)", "MARBERT"])

        if st.button("Analyze Sentiment"):
            if model_option == "BERT (Custom)":
                sentiment = predict_sentiment(st.session_state.transcribed_text)
            else:  # MARBERT
                sentiment = predict_sentiment_with_marbert(st.session_state.transcribed_text)

            st.write(f"Predicted Sentiment ({model_option}):")
            st.write(sentiment)

    # Section: Chatbot
    st.subheader("Chatbot")
    conversation = st.text_area("Conversation History:", st.session_state.transcribed_text, height=150)
    user_question = st.text_input("Ask a question in Arabic:")

    if st.button("Submit"):
        if user_question:
            response = ask_question(chat, st.session_state.transcribed_text, user_question)
            st.write("AI Response:")
            st.write(response)
            st.session_state.transcribed_text += f"\nسؤال: {user_question}\nجواب: {response}"
        else:
            st.warning("Please enter a question.")

# Customer dataset chat functionality
def customer_dataset_chat_page():

    st.markdown("""
    <style>
    /* Set black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Title and text styling */
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
        animation: fadeIn 1s ease-in;
    }

    /* Loading spinner styling */
    .stSpinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-top: 4px solid white;
        animation: spin 1s linear infinite;
    }

    /* Transcription and response box animation */
    .stTextArea, .stTextInput, .stButton {
        transition: all 0.3s ease;
    }
    .stTextArea:hover, .stTextInput:hover, .stButton:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 15px rgba(255, 255, 255, 0.5);
    }

    /* Fade-in animation for all elements */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    /* Spinner animation for loading indicator */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

    # CSS styling to modify the button appearance
    st.markdown("""
    <style>
    /* Set a global black background and white text */
    .main, .reportview-container {
        background-color: black;
        color: white;
    }

    /* Style for all button elements */
    .stButton > button {
        color: white !important; /* Ensures button text is white */
        font-size: 18px !important; /* Increases font size for better readability */
        font-weight: bold !important; /* Makes text bold */
        background-color: #333 !important; /* Dark button background */
        border-radius: 8px; /* Rounded corners for button */
        padding: 10px 20px; /* Adds padding to look like a custom button */
        transition: background-color 0.3s, transform 0.3s;
    }

    /* Hover effect for the button */
    .stButton > button:hover {
        background-color: #555 !important; /* Slightly lighter background on hover */
        box-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3); /* Adds a shadow on hover */
        transform: scale(1.05); /* Slight zoom-in effect on hover */
    }
    </style>
""", unsafe_allow_html=True)
    st.title("Chatting With All Customer Dataset")
    csv_file_path = 'datasets/cleaned_customer_calls.csv'
    df = pd.read_csv(csv_file_path)

    # Initialize the LLM wrapper and CSV agent
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash-8b",
        temperature=0,
        verbose=True,
        google_api_key="your_API_Key"
    )
    agent = create_csv_agent(llm, csv_file_path, verbose=False, allow_dangerous_code=True)

    # Initialize session state for chat history
    for key in ['history', 'generated', 'past']:
        if key not in st.session_state:
            st.session_state[key] = ["Hello! Ask me anything about the data 🤗"] if key == 'generated' else ["Hey! 👋"] if key == 'past' else []

    # Helper to check if text contains Arabic characters
    def has_arabic_chars(text):
        return any('\u0600' <= char <= '\u06FF' for char in text)

    def check_if_prompt_is_complex(response):
        # Convert the response to lowercase for case-insensitive matching
        word_list = ["compare", "between","draw", "chart", "table", "line", "make", "رسم", "مخطط", "ارسم", "اصنع", "قارن", "مقارنة", "جدول", "خط"]
        response = response.lower()

        # Iterate over the list of words
        for word in word_list:
            # If any word from the list is found in the response
            if word.lower() in response:
                return True
        return False

    # Query agent for response
    def query_agent_complex(agent, query):
        prompt = (
            """
                For the following query about customer call data, respond in a python format that can be used in Streamlit. Assuming 'df' is defined before examples:

                - If the query requires creating a line chart, reply with markdown Python code to generate and display it with Streamlit:
                  ```python
                  import pandas as pd
                  line_data = pd.DataFrame({"X_label": ["Label1", "Label2", ...], "Y_label": [value1, value2, ...]})
                  st.line_chart(line_data.set_index("X_label"))
                  ```

                - If the query requires drawing a table, reply with markdown Python code that generates a pandas DataFrame from the CSV and displays it with Streamlit example:
                  ```python
                  import pandas as pd
                  table_data = pd.DataFrame({"columns": ["column1", "column2", ...], "data": [["value1", "value2", ...], ["value1", "value2", ...], ...]})
                  st.write(table_data)
                  ```
                - If the query requires creating a bar chart, reply with markdown Python code to generate and display it with Streamlit:
                  ```python
                  import pandas as pd
                  bar_data = pd.DataFrame({"X_label": ["Label1", "Label2", ...], "Y_label": [value1, value2, ...]})
                  st.bar_chart(bar_data.set_index("X_label"))
                  ```

                Below is the query:
                Query:
                """
            + query
        )
        st.write("complex")
        return agent.run(prompt).__str__()

    def query_agent_simple(agent, query):
        prompt = (
            """
                - If the query is a general question, answer without Python code:
                  "answer"

                - If the query is unknown or cannot be answered, respond with:
                  "I do not know."

                  Below is the query:
                Query:
                """
            + query
        )
        st.write("simple")
        return agent.run(prompt).__str__()

    # Handle chat conversation
    def conversational_chat(query):
        if has_arabic_chars(query):
            query += "، وأجب باللغة العربية"
        try:
            if check_if_prompt_is_complex(query):
                response = query_agent_complex(agent, query)
                answer = "Please see the output below."
                return answer
            else:
                response = query_agent_simple(agent, query)
                return response
        except Exception as e:
            st.write(f"Error during query execution: {str(e)}")
            return "An error occurred while processing your request."

    # Containers for chat history and input
    response_container = st.container()
    container = st.container()

    # Chat input form
    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Ask something about your data here...")
            submit_button = st.form_submit_button(label='Send')
            if submit_button and user_input:
                answer = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(answer)

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i, (user_message, bot_message) in enumerate(zip(st.session_state['past'], st.session_state['generated'])):
                message(user_message, is_user=True, key=f"{i}_user", avatar_style="big-smile")
                message(bot_message, key=str(i), avatar_style="thumbs")


# Switch between pages based on login state
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login_page()
else:
    main_app_page()
