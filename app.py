from flask import Flask, redirect, url_for, render_template, request
import util

app = Flask(__name__)
app.secret_key = "hello"

all_corps = util.list_files_in_folder_no_extension('./static/reports')
all_perspectives = [
    'Perspective 1: Business',
    'Perspective 2: Properties',
    'Perspective 3: Legal Proceeding',
    'Perspective 4: Market for Registrant’s Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities',
    'Perspective 5: Management’s Discussion and Analysis of Financial Condition and Results of Operations',
    'Perspective 6: Financial Statements and Supplementary Data',
    'Perspective 7: Changes in and Disagreements With Accountants on Accounting and Financial Disclosure',
    'Perspective 8: Principal Accountant Fees and Services'
]
sources=['ChatGPT','Bart-GPU-Generating','LlaMA3.1-8B']
source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = '', '', '', '', '', '', '', '', '', '', '', ''

@app.route('/')
def home():
    util.initialization()
    return render_template('home.html')

@app.route('/introduction/')
def introduction():
    return render_template('introduction.html')

@app.route('/analysis/', methods=['POST', 'GET'])
def analysis():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    return render_template(
        'analysis.html',
        all_corps=all_corps,
        all_perspectives=all_perspectives,
        sources=sources,
        source=source,
        outputA=outputA,
        outputB=outputB,
        similarity_output_A=similarity_A,
        similarity_output_B=similarity_B,
        optionA=optionA,
        optionB=optionB,
        comparison=comparison,
        perspective=perspective,
        perspective_A=perspective_A,
        perspective_B=perspective_B,
        perspective_comparison=perspective_comparison,
    )

@app.route('/choose_summarization_source/', methods=['POST'])
def update_source():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    source = request.form['source']
    util.write_source(source)
    return redirect(url_for('analysis'))

@app.route('/update_selection_A/', methods=['POST'])
def update_selection_A():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    if source == "":
        return redirect(url_for('analysis'))
    optionA = request.form['optionA']
    util.write_optionA(optionA)
    file_path_A = f'./static/reports/{optionA}.json'
    summary_A = util.get_summarization(source, file_path_A)
    util.write_outputA(summary_A)
    return redirect(url_for('analysis'))

@app.route('/update_selection_B/', methods=['POST'])
def update_selection_B():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    optionB = request.form['optionB']
    util.write_optionB(optionB)
    file_path_B = f'./static/reports/{optionB}.json'
    summary_B = util.get_summarization(source, file_path_B)
    util.write_outputB(summary_B)
    return redirect(url_for('analysis'))

@app.route('/handle_similarity/', methods=['POST'])
def handle_similarity():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    file_path_A = f'./static/reports/{optionA}.json'
    summary_A = outputA
    similarity_output_A = util.handle_similarity(file_path_A, summary_A)
    similarity_A = similarity_output_A
    print(similarity_A)
    util.write_similarity_A(similarity_A)
    file_path_B = f'./static/reports/{optionB}.json'
    summary_B = outputB
    similarity_output_B = util.handle_similarity(file_path_B, summary_B)
    similarity_B = similarity_output_B
    util.write_similarity_B(similarity_B)
    return redirect(url_for('analysis'))

@app.route('/comparison/', methods=['POST'])
def analyst():
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    comparison = util.handle_analyst(outputA, outputB)
    util.write_comparison(comparison)
    return redirect(url_for('analysis'))

@app.route('/perspectives/', methods=['POST'])
def perspectives():
    perspective = request.form['perspective']
    util.write_perspective(perspective)
    source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison = util.read_cache()
    perspective_A, perspective_B, perspective_comparison = util.handle_perspective(perspective, outputA, outputB)
    util.write_perspective_A(perspective_A)
    util.write_perspective_B(perspective_B)
    util.write_perspective_comparison(perspective_comparison)
    return redirect(url_for('analysis'))

if __name__ == '__main__':
    app.run()
