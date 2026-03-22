import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import os
import tempfile
import pandas as pd
from inference import BanglaVQAPipeline

st.set_page_config(
    page_title='Bangla VQA',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem; font-weight: 700; color: #1f4e79;
        text-align: center; margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem; color: #555;
        text-align: center; margin-bottom: 2rem;
    }
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px; border-radius: 15px; text-align: center;
        color: white; font-size: 2rem; font-weight: bold; margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #667eea;
        padding: 10px 15px; border-radius: 5px; margin: 5px 0;
    }
    .predicted-card {
        background: #e8f5e9; border-left: 4px solid #4caf50;
        padding: 10px 15px; border-radius: 5px; margin: 5px 0;
    }
    .actual-card {
        background: #fff8e1; border-left: 4px solid #ffc107;
        padding: 10px 15px; border-radius: 5px; margin: 5px 0;
    }
    .correct-banner {
        background: #e8f5e9; border: 2px solid #4caf50; border-radius: 10px;
        padding: 12px; text-align: center; color: #1b5e20;
        font-weight: bold; font-size: 1.1rem; margin: 10px 0;
    }
    .incorrect-banner {
        background: #ffebee; border: 2px solid #ef5350; border-radius: 10px;
        padding: 12px; text-align: center; color: #b71c1c;
        font-weight: bold; font-size: 1.1rem; margin: 10px 0;
    }
    .footer {
        text-align: center; color: #aaa; font-size: 0.8rem;
        margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


#  Load model (cached)
@st.cache_resource(show_spinner='Loading Bangla VQA model...')
def load_pipeline():
    return BanglaVQAPipeline(
        checkpoint_path='checkpoints/best_model.pt',
        q_vocab_path='vocab/q_stoi.json',
        a_vocab_path='vocab/a_stoi.json',
        hidden_dim=512,
        fusion_type='concat',
        emb_dim=300,
        max_q_len=20,
        max_a_len=10,
        beam_size=3,
    )

pipeline = load_pipeline()


# Load test CSV for evaluation mode
@st.cache_data
def load_test_csv():
    for path in [
        'data/bangla_bayanno_test.csv',
        'bangla_bayanno_test.csv',
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns     = df.columns.str.strip().str.lower()
            df['image']    = df['image'].astype(str).str.strip()
            df['question'] = df['question'].astype(str).str.strip()
            df['answer']   = df['answer'].astype(str).str.strip()
            return df, path
    return None, None

test_df, csv_path = load_test_csv()


def lookup_actual(fname, question):
    if test_df is None: return ''
    fname_clean = os.path.basename(fname).strip()
    match = test_df[
        (test_df['image'] == fname_clean) &
        (test_df['question'] == question.strip())
    ]
    return str(match.iloc[0]['answer']) if not match.empty else ''


#  Result image
def create_result_image(image, question, predicted, actual=None):
    W     = 800
    img_h = int(image.height * W / image.width)
    img_r = image.resize((W, img_h))
    panel_h = 220 if actual else 160
    canvas  = Image.new('RGB', (W, img_h + panel_h), 'white')
    canvas.paste(img_r, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, img_h, W, img_h + panel_h], fill='#1f4e79')

    try:
        fl = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 22)
        fs = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    except Exception:
        try:
            fl = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
            fs = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        except Exception:
            fl = fs = ImageFont.load_default()

    y = img_h + 15
    q_d = question[:57] + '...' if len(question) > 60 else question
    draw.text((20, y), f'Q: {q_d}',              fill='#aed6f1', font=fs); y += 35
    draw.text((20, y), f'Predicted: {predicted}', fill='#a9dfbf', font=fl); y += 42
    if actual:
        draw.text((20, y), f'Actual:    {actual}', fill='#f9e79f', font=fl); y += 42
        mt = 'Correct' if predicted.strip() == actual.strip() else 'Incorrect'
        draw.text((20, y), mt, fill='#a9dfbf' if mt=='Correct' else '#f1948a', font=fs)
    buf = io.BytesIO()
    canvas.save(buf, format='PNG')
    return buf.getvalue()


# PDF
def create_pdf(image, question, predicted, actual=None):
    try:
        from fpdf import FPDF
        tmp = os.path.join(tempfile.gettempdir(), 'vqa_tmp.png')
        iw  = 800
        ih  = int(image.height * iw / image.width)
        image.resize((iw, ih)).save(tmp)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 15, 'Bangla VQA Result Report', ln=True, align='C')
        pdf.ln(2)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 8,
                 'Lightweight CNN-LSTM | EfficientNet-B0 + BiLSTM | ',
                 ln=True, align='C')
        pdf.ln(4)
        pdf.set_draw_color(102, 126, 234)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(7)

        piw = 170
        pih = min(int(ih * piw / iw), 110)
        pdf.image(tmp, x=20, w=piw, h=pih)
        pdf.ln(6)

        pdf.set_font('Helvetica', 'B', 13)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 10, 'Results', ln=True)
        pdf.ln(5)

        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(55, 9, 'Question:', ln=False)
        pdf.set_font('Helvetica', '', 11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 9, question)
        pdf.ln(3)

        pdf.set_fill_color(232, 245, 233)
        pdf.set_draw_color(76, 175, 80)
        pdf.set_line_width(0.5)
        pdf.rect(10, pdf.get_y(), 190, 16, 'DF')
        pdf.set_font('Helvetica', 'B', 11)
        pdf.set_text_color(27, 94, 32)
        pdf.cell(55, 16, '  Predicted Answer:', ln=False)
        pdf.set_font('Helvetica', 'B', 13)
        pdf.cell(0, 16, predicted, ln=True)
        pdf.ln(4)

        if actual:
            pdf.set_fill_color(255, 248, 225)
            pdf.set_draw_color(255, 193, 7)
            pdf.set_line_width(0.5)
            pdf.rect(10, pdf.get_y(), 190, 16, 'DF')
            pdf.set_font('Helvetica', 'B', 11)
            pdf.set_text_color(130, 90, 0)
            pdf.cell(55, 16, '  Actual Answer:', ln=False)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.cell(0, 16, actual, ln=True)
            pdf.ln(4)

            is_m = predicted.strip() == actual.strip()
            fc = (232,245,233) if is_m else (255,235,238)
            dc = (76,175,80)   if is_m else (239,83,80)
            tc = (27,94,32)    if is_m else (183,28,28)
            pdf.set_fill_color(*fc); pdf.set_draw_color(*dc)
            pdf.set_line_width(0.8)
            pdf.rect(10, pdf.get_y(), 190, 14, 'DF')
            pdf.set_font('Helvetica', 'B', 12)
            pdf.set_text_color(*tc)
            pdf.cell(0, 14, f'  {"Correct" if is_m else "Incorrect"}',
                     ln=True, align='C')

        try: os.remove(tmp)
        except: pass
        return bytes(pdf.output())
    except Exception:
        lines = ['BANGLA VQA RESULT REPORT', '='*40,
                 f'Question: {question}', f'Predicted: {predicted}']
        if actual: lines.append(f'Actual: {actual}')
        return '\n'.join(lines).encode('utf-8')


# Session state
for k, v in [('question_val',''),('history',[]),('last_result',None)]:
    if k not in st.session_state:
        st.session_state[k] = v

def set_question(q):
    st.session_state['question_val'] = q


# SIDEBAR
with st.sidebar:
    # st.image(
    #     'https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Flag_of_Bangladesh.svg/330px-Flag_of_Bangladesh.svg.png', width=100
    # )
    st.markdown('# About')
    st.markdown(
        'Lightweight **Bangla VQA** using **EfficientNet-B0 + BiLSTM**.\n\n'
        'Upload any image, ask a Bangla question, get a predicted answer.'
    )
    st.divider()

    st.markdown('### Model Performance')
    st.markdown('*(Bangla-Bayanno test set — 7,229 samples)*')
    c1, c2 = st.columns(2)
    c1.metric('Char-BLEU1', '0.4438')
    c2.metric('Exact Match', '37.29%')
    c1.metric('Word-BLEU1', '0.3599')
    c2.metric('F1',         '0.3818')
    st.caption('Beam Search k=3 | 16M params | 13ms/sample')

    st.divider()
    st.markdown('### Architecture')
    st.markdown(
        '- EfficientNet-B0 (last 2 blocks unfrozen)\n'
        '- BiLSTM question encoder\n'
        '- Question-Guided Spatial Attention\n'
        '- Concat Fusion\n'
        '- Autoregressive LSTM decoder\n'
        '- Beam Search (k=3)\n'
        '- Label Smoothing 0.1'
    )

    st.divider()
    st.markdown('# Best For')
    st.markdown(
        'Yes/No: `বাইরে কি ঠান্ডা?`\n\n'
        'Color: `এটি কোন রঙের?`\n\n'
        'Sport: `কোন খেলা চলছে?`\n\n'
        'Count: `কতজন মানুষ আছে?`'
    )

    st.divider()
    st.markdown('# Sample Questions')
    for q in ['বাইরে কি ঠান্ডা?','এটি কোন রঙের?','কোন খেলা চলছে?',
              'ছবিতে কতজন মানুষ আছে?','এটি কোথায়?','এটি কি করছে?']:
        st.button(q, key=f'sb_{q}', on_click=set_question,
                  args=(q,), use_container_width=True)

    st.divider()
    if test_df is not None:
        st.success(f'Eval CSV loaded: {len(test_df):,} samples')
        st.caption('Actual answers auto-fill when using test set images.')
    else:
        st.info('Place bangla_bayanno_test.csv in data/ for evaluation mode.')



# MAIN
st.markdown('<div class="main-title">Bangla Visual Question Answering</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Lightweight CNN-LSTM | EfficientNet-B0 + BiLSTM + Concat Fusion  '
    '</div>', unsafe_allow_html=True)

st.divider()

col1, col2 = st.columns([1, 1], gap='large')

with col1:
    st.markdown('### Upload Image')
    uploaded = st.file_uploader(
        'image', type=['jpg','jpeg','png','webp'],
        label_visibility='collapsed'
    )
    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        st.image(image, caption=uploaded.name, use_container_width=True)
        st.success(f'{uploaded.name} ({image.width}x{image.height}px)')
    else:
        st.info('Upload any image to get started')
        image = None

with col2:
    st.markdown('### Ask in Bangla')
    question = st.text_input(
        'question',
        value=st.session_state['question_val'],
        placeholder='যেমন: বাইরে কি ঠান্ডা?',
        label_visibility='collapsed',
    )
    st.session_state['question_val'] = question

    # Auto-lookup actual answer
    auto_actual = ''
    if uploaded and question and test_df is not None:
        auto_actual = lookup_actual(uploaded.name, question)

    st.markdown('**Actual Answer** *(optional — for evaluation)*')
    st.caption('Auto-fills if image+question found in test CSV. Or type manually.')
    actual_answer = st.text_input(
        'actual', value=auto_actual,
        placeholder='সঠিক উত্তর (ঐচ্ছিক)...',
        label_visibility='collapsed',
        key='actual_inp'
    )

    st.markdown('**Quick questions:**')
    qc = st.columns(2)
    for i, q in enumerate(['বাইরে কি ঠান্ডা?','এটি কোন রঙের?',
                            'কোন খেলা চলছে?','কতজন আছে?']):
        with qc[i % 2]:
            st.button(q, key=f'qq_{q}', on_click=set_question,
                      args=(q,), use_container_width=True)

    st.divider()
    get_answer = st.button(
        'Get Answer', type='primary',
        use_container_width=True,
        disabled=(image is None or not question)
    )
    if image is None:
        st.caption('Upload an image first')
    elif not question:
        st.caption('Type a question in Bangla')


# PREDICTION

st.divider()

if get_answer and image is not None and question:
    with st.spinner('Predicting...'):
        result = pipeline.predict(image, question)

    predicted = result['answer']
    actual    = actual_answer.strip() if actual_answer.strip() else None

    st.session_state['last_result'] = {
        'image':     image,
        'question':  question,
        'predicted': predicted,
        'actual':    actual,
        'tokens':    result.get('tokens', []),
        'fname':     uploaded.name if uploaded else '',
    }
    st.session_state['history'].append({
        'question':  question,
        'predicted': predicted,
        'actual':    actual or '—',
    })

# Display result
lr = st.session_state.get('last_result')
if lr:
    predicted = lr['predicted']
    actual    = lr['actual']
    question  = lr['question']
    image_lr  = lr['image']
    tokens    = lr.get('tokens', [])

    st.markdown(
        f'<div class="answer-box">{predicted}</div>',
        unsafe_allow_html=True
    )

    if actual:
        if predicted.strip() == actual.strip():
            st.markdown(
                f'<div class="correct-banner">Correct! '
                f'Predicted "{predicted}" = Actual "{actual}"</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="incorrect-banner">Incorrect — '
                f'Predicted: "{predicted}" | Actual: "{actual}"</div>',
                unsafe_allow_html=True
            )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><b>Question</b><br>{question}</div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="predicted-card"><b>Predicted</b><br>'
            f'<span style="font-size:1.3rem;font-weight:bold;'
            f'color:#1b5e20;">{predicted}</span></div>',
            unsafe_allow_html=True
        )
    with c3:
        if actual:
            st.markdown(
                f'<div class="actual-card"><b>Actual</b><br>'
                f'<span style="font-size:1.3rem;font-weight:bold;'
                f'color:#7d6608;">{actual}</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="metric-card"><b>Tokens</b><br>'
                f'{" | ".join(tokens) if tokens else "N/A"}</div>',
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown('### Download')
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            'Download Result Image',
            data=create_result_image(image_lr, question, predicted, actual),
            file_name='bangla_vqa_result.png', mime='image/png',
            use_container_width=True
        )
    with d2:
        st.download_button(
            'Download PDF Report',
            data=create_pdf(image_lr, question, predicted, actual),
            file_name='bangla_vqa_report.pdf', mime='application/pdf',
            use_container_width=True
        )

# History
if st.session_state['history']:
    st.divider()
    st.markdown('### Recent Questions')
    for h in reversed(st.session_state['history'][-5:]):
        icon = ''
        if h['actual'] != '—':
            icon = ' OK' if h['predicted'].strip()==h['actual'].strip() else ' X'
        st.markdown(
            f'**Q:** {h["question"]} -> '
            f'**Pred:** `{h["predicted"]}` | '
            f'**Actual:** `{h["actual"]}`{icon}'
        )

st.markdown(
    '<div class="footer">'
    'Lightweight Bangla VQA | EfficientNet-B0 + BiLSTM + Concat Fusion | '
    '</div>', unsafe_allow_html=True
)