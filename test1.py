#필요 라이브러리 모음
#sudo apt update
#sudo apt install ffmpeg
#uv add pydub
#uv add kagglehub
#uv add SpeechRecognition
#uv add transformers
#uv add langchain
#uv add langchain_community
#uv add langchain_openai
#uv add sentence-transformers
#uv add faiss-gpu
#uv add pandas
#uv add bs4
#uv add -U langsmith
#uv add accelerate
#uv add gtts


# mp3 파일을 wav 파일로 변환, 이미 wav로 파일이 저장되어 있다면, 생략 가능
from pydub import AudioSegment
import os
import kagglehub

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio.export(wav_file_path, format="wav")

# 예시: 디렉토리 내의 모든 .mp3 파일을 .wav 파일로 변환
#path = kagglehub.dataset_download("wiradkp/mini-speech-diarization")
#directory_path = f"{path}/dataset/raw"
#wav_directory_path = "./converted_wav"

# 예시2: input 로컬 폴더에서 mp3 파일을 불러와서 변환
directory_path = "/home/minho-song/stt_test_1/input"
wav_directory_path = "./converted_wav"


os.makedirs(wav_directory_path, exist_ok=True)

for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".mp3"):
        mp3_file_path = os.path.join(directory_path, filename)
        wav_file_path = os.path.join(wav_directory_path, filename.replace(".mp3", ".wav"))
        convert_mp3_to_wav(mp3_file_path, wav_file_path)
        print(f"Converted {mp3_file_path} to {wav_file_path}")


import speech_recognition as sr
#import os
import pandas as pd

def speech_to_text_from_audio(audio_file_path, language="ko-KR", chunk_length=60000):
    recognizer = sr.Recognizer()
    
    audio = AudioSegment.from_wav(audio_file_path)
    total_duration = len(audio)
    texts = []

    for i in range(0, total_duration, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk.export("temp.wav", format="wav")
        with sr.AudioFile("temp.wav") as source:
            audio_chunk = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_chunk, language=language)
            texts.append(text)
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand audio from {audio_file_path} at {i} ms")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    return " ".join(texts)

# 경로 설정
# 파일경로설정 / 해당 파일안에 있는 모든 wav 파일에 대해 stt를 수행할 거임.
directory_path = "/home/minho-song/stt_test_1/converted_wav"

text = []
# 디렉토리 내의 모든 .wav 파일에 대해 STT 수행
for filename in sorted(os.listdir(directory_path)):
    if filename.endswith(".wav"):
        audio_file_path = os.path.join(directory_path, filename)
        text.append(speech_to_text_from_audio(audio_file_path))

#결과 확인용.
print(text)


import bs4
from langchain_classic import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.docstore.document import Document
from langchain_classic.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_classic.chains import RetrievalQA
from langchain_classic.document_loaders import TextLoader
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.llms import HuggingFacePipeline
from transformers import RobertaForCausalLM, RobertaTokenizer, pipeline


# 문서를 지정된 크기의 청크로 나눔.
doc = Document(page_content=text[0]) # 이때 text가 리스트 형식이면 안됨, 리스트 형식이면 document 형태로 변환 불가임
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

splits = text_splitter.split_documents([doc])

#확인용
len(splits)
print(splits)


from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration, pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_classic.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 

gamma_path = kagglehub.model_download("google/gemma-2/transformers/gemma-2-2b-it/1")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

retriever = vectorstore.as_retriever()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", use_auth_token="HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    use_auth_token="HF_TOKEN",
    device_map="auto")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device) # <- 오류 시 주석 처리 후 .input_ids.to('cpu') 주석 해제

def generate_answer(question):
    related_docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in related_docs])
    input_text = f"{context}\n\nQuestion: {question}\nAnswer:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to('cpu') # model.to(device) <- 오류 시 주석 해제.
    generated_ids = model.generate(input_ids, max_length=1024, num_return_sequences=1)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 질문을 입력받아 답변 생성
question = "이 텍스트의 주요 내용은 무엇인가요?"
answer = generate_answer(question)
print(answer)

def extract_answer(text):
    # '\nAnswer:' 이후의 텍스트를 추출
    if '\nAnswer:' in text:
        answer = text.split('\nAnswer:')[1].strip()
    else:
        answer = "Answer 부분을 찾을 수 없습니다."
    
    return answer

# Answer 부분만 추출
only_answer = extract_answer(answer)
print(only_answer)


from gtts import gTTS
#import os

def text_to_speech(text, language='ko', output_file='output.mp3'):
    """
    텍스트를 음성으로 변환하여 mp3 파일로 저장합니다.

    Parameters:
    text (str): 변환할 텍스트
    language (str): 음성 언어 (기본값은 영어 'en')
    output_file (str): 저장할 mp3 파일 이름 (기본값은 'output.mp3')
    """
    # gTTS 객체 생성
    tts = gTTS(text=text, lang=language, slow=False)

    # 음성 파일 저장
    tts.save(output_file)
    print(f"음성 파일이 {output_file}로 저장되었습니다.")

    # 음성 파일 재생 (옵션)
    os.system(f"start {output_file}")  # Windows에서 작동
    # os.system(f"afplay {output_file}")  # macOS에서 작동
    # os.system(f"mpg321 {output_file}")  # Linux에서 작동

# 예제 사용
text_to_speech(only_answer, language='ko', output_file='hello.mp3')