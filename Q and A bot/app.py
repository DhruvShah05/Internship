from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import sqlite3
import google.generativeai as genai
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key

def print_schema(db_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Execute the schema query
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
    
    # Fetch all results
    schemas = cursor.fetchall()
    
    # Print the schema of each table
    for schema in schemas:
        print(schema[0])
    
    # Close the connection
    conn.close()

# Specify the path to your SQLite database
database_path = 'database.db'

# Print the schema
print_schema(database_path)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure the API key
os.environ["GEMINI_API_KEY"] = "AIzaSyA-gjN62gU-AKjaj56VSgSbSZqVZAMSE8o"  # Replace with your actual Gemini API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
generation_config = {
    "temperature": 0.1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Function to load the CSV file
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)

# Function to convert CSV to SQLite
def csv_to_sqlite(file_path):
    try:
        df = pd.read_csv(file_path)
        conn = sqlite3.connect('database.db')
        df.to_sql('data', conn, if_exists='replace', index=False)
        conn.close()
    except Exception as e:
        return str(e)
    return None

# Function to generate dataset summary
def generate_dataset_summary(df):
    summary = {
        "table_name": "data",
        "columns": df.columns.tolist(),
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "sample_data": df.head().to_dict(orient='records')
    }
    return summary

# Function to execute SQL query or get schema information
def execute_sql(query):
    conn = sqlite3.connect('database.db')
    try:
        if query.lower().startswith('pragma'):
            cursor = conn.execute(query)
            columns = [description[0] for description in cursor.description]
            result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            return result, None
        else:
            result = pd.read_sql_query(query, conn)
            return result.to_dict(orient='records'), None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

# Function to generate SQL or non-SQL answer using the Gemini model
def generate_answer(query, summary):
    context = (
        f"The dataset is stored in a table named '{summary['table_name']}'. "
        f"It has columns: {summary['columns']}. "
        f"Rows: {summary['num_rows']}, Columns: {summary['num_columns']}. "
        f"Sample data: {summary['sample_data'][:1]}. "
        f"Answer the question or convert it into a clean SQL query suitable for execution directly, using the actual table and column names, without any additional explanation or formatting artifacts:\n\n"
        f"{query}\n\n"
        f"If the answer is not SQL-based, provide a concise and informative response, limited to a few sentences. Return only the necessary content without additional formatting or explanations."
    )
    
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(context)
    
    # Check for SQL keywords to identify if the response is an SQL query
    if "SELECT" in response.text and "FROM" in response.text or "PRAGMA" in response.text:
        sql_query = response.text.strip()
        # Remove any formatting artifacts such as code block syntax
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        return sql_query, "sql"
    else:
        return response.text.strip(), "non_sql"

# Function to answer the query
def answer_query(query):
    summary = session.get('dataset_summary')
    response, query_type = generate_answer(query, summary)
    print_schema(database_path)
    print(response)
    if query_type == "sql":
        result, error = execute_sql(response)
        if error:
            return f"Error executing SQL query: {error}"
        return result
    else:
        return response

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='Please select a file.')
            if not allowed_file(file.filename):
                return render_template('index.html', error='Unsupported file type. Please upload a CSV file.')
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            session['file_path'] = file_path  # Save file path in session

            df, error = load_csv(file_path)
            if error:
                return render_template('index.html', error=f'Error loading CSV: {error}')
            
            error = csv_to_sqlite(file_path)
            if error:
                return render_template('index.html', error=f'Error converting CSV to SQLite: {error}')

            summary = generate_dataset_summary(df)
            session['dataset_summary'] = summary  # Save summary in session

            return redirect(url_for('query'))
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def query():
    if 'file_path' not in session:
        return redirect(url_for('index'))
    
    query = None
    answer = None

    if request.method == 'POST':
        query = request.form['query']
        answer = answer_query(query)
    
    return render_template('query.html', query=query, answer=answer)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)
