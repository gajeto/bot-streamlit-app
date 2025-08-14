# 🤖 Bot Agent – Streamlit Deployment

A conversational AI bot agent deployed with [Streamlit](https://streamlit.io/).  
Supports real-time interaction, custom prompts, and easy cloud deployment.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/username/bot-agent-streamlit.git
cd bot-agent-streamlit

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Locally

```bash
streamlit run app.py
```

The app will be available at:  
**http://localhost:8501**

---

## ⚙️ Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4
```

---

## 📂 Project Structure

```
bot-agent-streamlit/
│── app.py                # Main Streamlit app
│── bot_logic.py          # Bot agent logic
│── requirements.txt      # Python dependencies
│── .env                  # Environment variables
│── README.md             # Project documentation
```

---

## 🌐 Deployment

### Streamlit Cloud
1. Push your repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Deploy directly from your GitHub repository.
4. Add your `.env` secrets in the Streamlit dashboard.

---

## 📜 License
This project is licensed under the MIT License.
