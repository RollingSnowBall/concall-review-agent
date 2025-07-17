import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss  # FAISS 임포트


# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정 및 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.")
client = OpenAI(api_key=api_key)

# NLTK 데이터 다운로드
#nltk.download('punkt')
#nltk.download('punkt_tab')

# 1. TXT 파일 읽기 및 Semantic Chunking
def semantic_chunking(file_path, max_chunk_size=500, similarity_threshold=0.7):
    # TXT 파일 읽기
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # 문장 단위로 분할
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []
    
    # 임시 임베딩 모델 로드 (Semantic Chunking용)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Semantic Chunking: 유사한 문장 그룹화
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = sentence_embeddings[0].unsqueeze(0)
    
    for i in range(1, len(sentences)):
        # 현재 문장과 이전 청크의 유사성 계산
        similarity = util.cos_sim(sentence_embeddings[i], current_embedding[-1]).item()
        
        # 유사성이 임계값 이상이거나, 청크 크기가 작으면 추가
        if similarity >= similarity_threshold and len(' '.join(current_chunk)) + len(sentences[i]) <= max_chunk_size:
            current_chunk.append(sentences[i])
            current_embedding = sentence_embeddings[:i+1].mean(dim=0).unsqueeze(0)
        else:
            # 새로운 청크 시작
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i]]
            current_embedding = sentence_embeddings[i].unsqueeze(0)
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# 2. 임베딩 (OpenAI text-embedding-ada-002)
def embed_chunks(chunks):
    try: 
        response = client.embeddings.create(input=chunks, model="text-embedding-ada-002")
        return [choice.embedding for choice in response.data]
    except Exception as e:
        print(f"임베딩 중 에러 발생: {e}")
        return []
    
# 3. 벡터 DB 생성 (FAISS)
def create_vector_db(embeddings):
    if not embeddings:
        return None
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype='float32'))
    return index

# 4. 질문 번역 (한국어 → 영어)
def translate_question(korean_question):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "This is about Conference Call. user will ask about that. Translate the following Korean question to English."},
                {"role": "user", "content": korean_question}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"질문 번역 중 에러 발생: {e}")
        return korean_question  # 번역 실패 시 원문 반환

# 5. RAG: 질문으로 관련 청크 검색
def retrieve_chunks(query, index, chunks, k=10):
    try:
        query_embedding = client.embeddings.create(input=[query], model="text-embedding-ada-002").data[0].embedding
        D, I = index.search(np.array([query_embedding], dtype='float32'), k)
        return [chunks[i] for i in I[0]]
    except Exception as e:
        print(f"검색 중 에러 발생: {e}")
        return []
    
# 6. LLM 질의: 한국어 답변 생성
def generate_response(korean_question, english_question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""
    Context: {context}
    Question: {english_question}
    Instruction: Analyze the context and answer the question in Korean. Provide a concise summary and reasons for the analysis, including key phrases from the context.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM 질의 중 에러 발생: {e}")
        return "답변 생성 실패."


# 메인 실행
def main():
    file_path = "./input-document/pltr_earning_call_2024_1Q.txt"  # 네 TXT 파일 경로
    
    # 청킹
    try:
        chunks = semantic_chunking(file_path)
    except Exception as e:
        print(f"청킹 중 에러 발생: {e}")
        return
    
    # 임베딩
    embeddings = embed_chunks(chunks)
    if not embeddings:
        print("임베딩 생성 실패. API 키 또는 네트워크 확인.")
        return
    
    # 벡터 DB 생성
    index = create_vector_db(embeddings)
    if not index:
        print("벡터 DB 생성 실패.")
        return
    
    # 반복 질문 처리
    while True:
        korean_question = input("질문을 입력하세요 (종료하려면 '종료' 입력): ")
        if korean_question.strip() == "종료":
            print("프로그램 종료.")
            break
        
        english_question = translate_question(korean_question)
        retrieved_chunks = retrieve_chunks(english_question, index, chunks)
        if not retrieved_chunks:
            print("관련 청크 검색 실패.")
            continue
        
        response = generate_response(korean_question, english_question, retrieved_chunks)
        print("답변:", response)

if __name__ == "__main__":
    main()