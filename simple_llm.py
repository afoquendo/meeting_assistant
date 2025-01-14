from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

my_credentials = {
    "url"    : "https://us-south.ml.cloud.ibm.com"
}

params = {
        GenParams.MAX_NEW_TOKENS: 800, # The maximum number of tokens that the model can generate in a single run.
        GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
    }

LLAMA2_model = Model(
        model_id= 'meta-llama/llama-3-8b-instruct', 
        credentials=my_credentials,
        params=params,
        project_id="skills-network",  
        )

llm = WatsonxLLM(LLAMA2_model)  

#######------------- Prompt Template-------------####
# This template is structured based on LLAMA2. If you are using other LLMs, feel free to remove the tags
temp = """
List the key points with details from the context (maximum 10): 
The context : {context}
"""

pt = PromptTemplate(
    input_variables=["context"],
    template= temp)

prompt_to_LLAMA2 = LLMChain(llm=llm, prompt=pt)

def summarize_text(text):       
   return prompt_to_LLAMA2.run(text) 


