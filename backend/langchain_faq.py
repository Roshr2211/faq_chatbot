import os
import sys
import json
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = api_key

# Define the FAQ content
faq_content = """
1. What is silk?
Silk is a continuous protein filament secreted by specific types of caterpillars known as 'Silk Worms'.

2. What are the properties of silk?
Silk has best properties like, drape, light weight, Natural sheen, inherent affinity for rich colors, high moisture absorbance, resilience and excellent drape making it the ‘Queen of textiles’.

3. What are different types of silks available in India?
Different types of silks produced in India are Mulberry Silk (Karnataka, Andhra Pradesh, Tamil Nadu, J&K) Tassar Silk (Chattisghar, Madhyapradesh, Maharashtra, Telangana, West Bengal), Eri Silk (Assam and other N Eastern states) & Muga Silk (Assam).

4. Can we identify silk by a feel of touch?
It is very difficult to identify silk just by look and feel.

5. What is the best way to identify silk?
Silk is a protein fiber. When we closely observe, the smell of a burnt thread of silk, it is like smell of a burnt hair. In short, both burnt silk and burnt human hair have same smell.

6. What is Silk Mark label?
Silk Mark label is affixed on only 100% pure silk products by Silk Mark Authorised Users to protect the interest of silk consumers.

7. How to identify Silk Mark label?
Please watch https://youtu.be/Mq10TkRZTcY

8. What is the price of Silk Mark labels?
At present the price of one Silk Mark label is ₹5.90 (incl. of taxes)

9. How do consumers benefit by Silk Mark?
By purchasing Silk Mark labeled products from the sellers who are Authorised Users of Silk Mark, Consumers get an assurance of purchasing 100% Natural silk products.

10. What are the benefits of becoming a Silk Mark member?
1. Can gain the confidence of elite silk consumers
2. Cost effective publicity through SMOI
3. Can participate in Silk Mark expos across the country

11. How to become a Silk Mark member?
Visit www.silkmarkindia.com or contact nearest Silk Mark office

12. What if Silk Mark label is misused?
If an Authorised User sells non-silk products as 100% Natural Silk with Silk Mark label, initially notice will be given and if repeated the membership will be cancelled. The firm or company will be notified in leading newspapers about the misuse.

13. How are consumers protected in case of misuse of Silk Mark labels?
SMOI will facilitate the consumer in replacement or refund and also in pursuing legal redressal in consumer courts.

14. Can the Silk Mark label be duplicated?
For a given safety measures like Hologram, Unique Serial Number and QR Code, it is very difficult to make duplicate Silk Mark labels.

15. Does Silk Mark assure the purity/quality of zari?
No, Silk Mark assures only products with 100% pure silk.

16. Where do we get Silk Mark labeled products?
Silk Mark authorized weavers, manufacturers, retailers, showrooms and boutiques sell Silk Mark labeled products.

17. Where do we get 100% pure and genuine natural silk products?
Visit Silk Mark authorized outlets.

18. Where is the shop of Silk Mark organisation of India?
Silk Mark Organisation of India neither manufactures nor sells any silk products. SMOI does not have its own showroom or shop.

19. How to search for Silk Mark authorized outlets?
One can visit www.silkmarkindia.com and click on ‘authorized outlets’ button to search weavers, manufacturers, retailers, showrooms & boutiques selling 100% pure silk products with Silk Mark assurance.

20. Can we buy 100% pure silks online?
Yes. One can buy 100% pure silks online. However, ensure presence of Silk Mark labels on the products and collect the original bill/receipt.

21. Can we participate in the Silk Mark exhibitions?
Yes, an Authorized User of Silk Mark can participate in the Silk Mark Exhibitions across the country. However, stalls will be allotted on payment of prescribed Stall Rent.

22. Can we get our silk material tested for purity?
Yes, at SMOI offices and silk testing centers across the country.

23. Where can we get our silk material tested for purity?
Please visit silk testing centers/ Silk Mark offices detailed in www.silkmarkindia.com.

24. What is the lifespan of a silk sari?
If advised care is taken in a proper way, a silk sari can last for generations.
"""

# Split the text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = text_splitter.split_text(faq_content)

# Generate embeddings
embeddings = OpenAIEmbeddings()

# Create a FAISS vector store from the texts and embeddings
docsearch = FAISS.from_texts(texts, embeddings)

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

chain = load_qa_chain(llm, chain_type="stuff")

def answer_question(query):
    # Perform similarity search
    docs = docsearch.similarity_search(query)
    # Run the chain to get the answer
    response = chain.run(input_documents=docs, question=query)
    return response

if __name__ == "__main__":
    question = sys.argv[1]
    answer = answer_question(question)
    print(json.dumps({"answer": answer}))
