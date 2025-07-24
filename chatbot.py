import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        # Load document
        if filename.lower().endswith('.pdf'):
            loader = UnstructuredPDFLoader(path)
        else:
            loader = TextLoader(path, encoding='utf8')
        docs = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        # Embed and index
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        index_path = path + '.faiss'
        vectorstore.save_local(index_path)

        return jsonify({ 'index_path': index_path })
    return "Invalid file type", 400

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    index_path = data.get('index_path')
    if not question or not index_path:
        return "Missing parameters", 400

    # Load vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings)

    # Create QA pipeline
    llm = ChatOpenAI(model_name='gpt-4', temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(search_kwargs={'k': 4}))

    # Get answer
    answer = qa.run(question)
    return jsonify({ 'answer': answer })

if __name__ == '__main__':
    app.run(debug=True)