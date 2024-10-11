import os
from dotenv import load_dotenv
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

load_dotenv()

df = pd.read_csv('fidelizacao.csv', sep=';')
df = df.drop('OBSERVAÇÃO', axis=1)
df_resumo = df.groupby(['TURNO', 'STATUS']).size()
df_text = df_resumo.to_string()
#df_text = df.to_string(index=False)

template = """
Você é um analista de dados especializado em informações educacionais. Vou fornecer um DataFrame contendo dados de alunos com as seguintes colunas: ALUNO, MATRICULA, SEGMENTO, SERIE, TURNO, TURMA, e STATUS. Sua tarefa é analisar esses dados e responder às perguntas que eu fizer, sempre baseando suas respostas no DataFrame. Suas respostas devem ser claras, objetivas e orientadas por dados. Quando necessário, ofereça resumos, estatísticas e insights relevantes.

Aqui está uma breve descrição de cada coluna do DataFrame:
ALUNO: Nome completo do aluno.
MATRICULA: Número de matrícula do aluno.
SEGMENTO: Fase educacional do aluno, como 'EI' (Educação Infantil), 'EF' (Ensino Fundamental), 'EM' (Ensino Médio). Pode descopnsiderar o número, já que ele vem respresentado na série. Outra observação, do EF1 ao EF5 você vai considerar Fundamental 1 e do EF6 ao EF9 você vai considerar Fundamental 2.
SERIE: Série em que o aluno está matriculado.
TURNO: Período do dia em que o aluno estuda, como 'M' (Manhã) ou 'T' (Tarde).
TURMA: Turma à qual o aluno pertence.
STATUS: Status de matrícula, podendo ser 'Fidelizado' ou 'Matriculado'.

O DataFrame com os dados já está abaixo. Com base nesses dados, responda às minhas perguntas. 

Aqui estão os dados:
{df}

Exemplos de perguntas que eu posso fazer:
Quantos alunos estão matriculados em cada SEGMENTO?
Quantos alunos estão Fidelizados em cada TURNO?
Qual é a SÉRIE com mais alunos matriculados no período da TARDE?
Quantos alunos há por TURMA em determinado SEGMENTO?
Quais alunos estão com o status de 'Matriculado'?
Quantos alunos temos no total?

Sempre que possível, me apresente os resultados de forma sumarizada e, se houver informações que possam melhorar a análise, como proporções ou insights adicionais, inclua-os.
"""

prompt = PromptTemplate.from_template(template=template)
chat = ChatGroq(model='llama3-70b-8192', temperature=0.5)
chain = prompt | chat

# input_data = {
#     "df": df_text,
#     "question": "Quantos alunos estão matriculados no total?"
# }

# resposta = chain.invoke(input_data)
# print(resposta)
# Passar os dados como um dicionário e incluir o DataFrame diretamente no prompt
input_data = {
    "df": df_text
}

# A sua pergunta vai aqui
pergunta = "Quantos alunos estão com status de matriculado no total?"

# Invocar o modelo com a pergunta e os dados
response = chain.invoke(f"{pergunta}\n\nAqui estão os dados: {df_text}")

# Imprimir a resposta do modelo
print(response.content)